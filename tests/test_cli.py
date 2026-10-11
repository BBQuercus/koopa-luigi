"""Regression tests for final-task failures and CLI success reporting."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import Mock

import luigi
from luigi.execution_summary import LuigiStatusCode
import pytest

from src import cli
from src.util import FileStatusTracker, set_logging as real_set_logging


@pytest.fixture
def pipeline(tmp_path, monkeypatch, caplog):
    loggers = [logging.getLogger()] + [
        logger for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    logging_state = [
        (logger, logger.level, list(logger.handlers), logger.propagate)
        for logger in loggers
    ]
    args = SimpleNamespace(
        config="unused.cfg", env_info=False, gpu=False, workers=8,
        verbose=False, skip_incompatible=False,
    )
    config = {
        "input_path": str(tmp_path), "output_path": str(tmp_path),
        "file_ext": "companion.ome",
    }
    tracker = FileStatusTracker()
    monkeypatch.setattr(cli, "file_tracker", tracker)
    # Isolate Luigi's process-wide config and preserve pytest's log capture.
    luigi_config = luigi.configuration.LuigiConfigParser()
    luigi_config.add_section("core")
    monkeypatch.setattr(luigi.configuration, "get_config", lambda: luigi_config)
    monkeypatch.setattr(cli.cli_args, "_parse_args", lambda: args)
    monkeypatch.setattr(cli.util, "set_configuration", lambda path: None)
    monkeypatch.setattr(cli.util, "get_configuration", lambda: config)
    monkeypatch.setattr(cli.util, "set_logging", lambda **kwargs: None)
    monkeypatch.setattr(cli.util, "get_environment_info", lambda: {
        "python_version": "test", "virtual_env": None,
    })
    monkeypatch.setattr(cli, "_suppress_keras_progress", lambda: None)
    monkeypatch.setattr(cli, "run_preflight", lambda config: None)
    caplog.set_level(logging.INFO)

    targets = [luigi.LocalTarget(str(tmp_path / name)) for name in (
        "summary.csv", "summary_cells.csv",
    )]
    task = SimpleNamespace(output=lambda: targets)
    cpu_task = Mock(return_value=task)
    gpu_task = Mock(return_value=task)
    monkeypatch.setattr(cli, "Merge", cpu_task)
    monkeypatch.setattr(cli, "GPUMerge", gpu_task)
    result = SimpleNamespace(
        status=LuigiStatusCode.SUCCESS,
        summary_text="Luigi Execution Summary: final task did not run",
    )

    def build(tasks, **kwargs):
        tracker.register_file("image")
        tracker.mark_success("image")
        return result

    build_mock = Mock(side_effect=build)
    monkeypatch.setattr(luigi, "build", build_mock)

    def write_outputs():
        for target in targets:
            with target.open("w") as stream:
                stream.write("value\n1\n")

    yield SimpleNamespace(
        args=args, config=config, tracker=tracker, targets=targets,
        task=task, result=result, build=build_mock, write_outputs=write_outputs,
        cpu_task=cpu_task, gpu_task=gpu_task, luigi_config=luigi_config,
    )
    tracker.reset()
    for logger, level, handlers, propagate in logging_state:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.mark.parametrize("status", [
    status for status in LuigiStatusCode
    if status not in (LuigiStatusCode.SUCCESS, LuigiStatusCode.SUCCESS_WITH_RETRY)
])
def test_failed_build_exits_nonzero_even_with_outputs(pipeline, caplog, status):
    pipeline.write_outputs()
    pipeline.result.status = status
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert status.name in caplog.text
    assert pipeline.result.summary_text in caplog.text
    assert "files processed successfully" not in caplog.text
    assert cli.BANNER_SUCCESS.strip().splitlines()[0] not in caplog.text


@pytest.mark.parametrize("missing", [0, 1, "both"])
def test_missing_summaries_fail_even_if_luigi_and_files_succeeded(
    pipeline, caplog, missing,
):
    pipeline.write_outputs()
    for index, target in enumerate(pipeline.targets):
        if missing == "both" or missing == index:
            os.remove(target.path)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "Final pipeline output not produced" in caplog.text
    assert pipeline.result.summary_text in caplog.text
    assert "files processed successfully" not in caplog.text


@pytest.mark.parametrize("status", [
    LuigiStatusCode.SUCCESS, LuigiStatusCode.SUCCESS_WITH_RETRY,
])
def test_success_accepts_existing_outputs(pipeline, caplog, status):
    pipeline.write_outputs()
    for target in pipeline.targets:
        os.utime(target.path, (1, 1))
    pipeline.result.status = status
    cli.main()
    assert cli.BANNER_SUCCESS.strip().splitlines()[0] in caplog.text
    assert pipeline.build.call_args.kwargs["detailed_summary"] is True
    assert pipeline.build.call_args.kwargs["workers"] == 8
    assert pipeline.build.call_args.kwargs["log_level"] == "WARNING"
    assert pipeline.luigi_config.getboolean("core", "no_configure_logging")


@pytest.mark.parametrize("complete", [True, False])
def test_gpu_checks_marker_without_requiring_csvs(pipeline, caplog, complete):
    pipeline.args.gpu = True
    pipeline.targets[:] = [luigi.LocalTarget(
        os.path.join(pipeline.config["output_path"], ".gpu_complete")
    )]
    if complete:
        pipeline.write_outputs()
        cli.main()
        assert cli.BANNER_SUCCESS.strip().splitlines()[0] in caplog.text
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
        assert ".gpu_complete" in caplog.text
    assert pipeline.build.call_args.kwargs["workers"] == 1
    pipeline.gpu_task.assert_called_once()
    pipeline.cpu_task.assert_not_called()


def test_build_exception_records_traceback(pipeline, caplog):
    pipeline.build.side_effect = RuntimeError("scheduler unavailable")
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "scheduler unavailable" in caplog.text
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("file_status", ["failed", "processing", "pending", "none"])
def test_incomplete_files_fail_even_with_successful_build_and_outputs(
    pipeline, caplog, file_status,
):
    pipeline.write_outputs()

    def build(tasks, **kwargs):
        if file_status != "none":
            pipeline.tracker.register_file("image")
        if file_status == "failed":
            pipeline.tracker.mark_failed("image", "bad data")
        elif file_status == "processing":
            pipeline.tracker.mark_processing("image")
        return pipeline.result

    pipeline.build.side_effect = build
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert "files processed successfully" not in caplog.text


class FilePrerequisite(luigi.Task):
    directory = luigi.Parameter()
    index = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.directory, f"file-{self.index}"))

    def run(self):
        with self.output().open("w") as stream:
            stream.write("processed")


class FinalTask(luigi.Task):
    directory = luigi.Parameter()
    fail = luigi.BoolParameter()

    def requires(self):
        return [FilePrerequisite(directory=self.directory, index=i) for i in range(8)]

    def output(self):
        return [luigi.LocalTarget(os.path.join(self.directory, name)) for name in (
            "summary.csv", "summary_cells.csv",
        )]

    def run(self):
        if self.fail:
            raise RuntimeError("final merge failed after all files completed")
        for target in self.output():
            with target.open("w") as stream:
                stream.write("value\n1\n")


# Keep the real scheduler entry point before the fixture replaces it.
real_build = luigi.build


@pytest.mark.parametrize("fail", [False, True])
def test_real_eight_worker_build_reports_final_task_outcome(
    pipeline, monkeypatch, caplog, fail,
):
    def final_task(**kwargs):
        for index in range(8):
            pipeline.tracker.register_file(f"image-{index}", already_complete=True)
        return FinalTask(directory=pipeline.config["output_path"], fail=fail)

    monkeypatch.setattr(cli, "Merge", final_task)
    monkeypatch.setattr(luigi, "build", real_build)
    monkeypatch.setattr(cli.util, "set_logging", real_set_logging)
    if fail:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
        log_text = (pipeline.config["output_path"] + "/koopa.log")
        with open(log_text) as stream:
            log_text = stream.read()
        assert "Luigi Execution Summary" in log_text
        assert "FinalTask" in log_text
        assert "Traceback" in log_text
        assert "final merge failed after all files completed" in log_text
        assert "files processed successfully" not in log_text
        assert not any(target.exists() for target in pipeline.targets)
        assert all(task.complete() for task in final_task().requires())
    else:
        cli.main()
        assert all(target.exists() for target in pipeline.targets)
        # A second invocation reuses Luigi's completed outputs successfully.
        cli.main()
        with open(pipeline.config["output_path"] + "/koopa.log") as stream:
            assert cli.BANNER_SUCCESS.strip().splitlines()[0] in stream.read()
