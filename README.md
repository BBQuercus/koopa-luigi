# koopa-luigi
A functional implementation of koopa in luigi :)

Notes:
* If creating a local installation, install with `pip install -e .`
* Everything will be run via `koopa-luigi` (the `.sh` files just abstract that)
* GPU workflow only triggeres GPU-dependent components and does not merge
* CPU / standard workflow will trigger everything

Usage:
* Download the `koopa.cfg` and `cpu/gpu.sh` files somewhere server accessible
* Update `koopa.cfg` as usual
* Update the `cpu/gpu.sh` files:
	* Change the `CONFIG` parameter to the path to `koopa.cfg`
	* Add a mail account if running on slurm
	* Change resources if running on slurm
* Log into slurm or xenon servers
* Running on slurm:
	* Run either the GPU or CPU workflow separately depending on requirements with `sbatch NAME.sh`
	* The CPU workflow is optionally dependent on the GPU workflow
	* Trigger the GPU workflow with `my_id=$(sbatch --parsable NAME.sh)`
	* Add the CPU workflow as dependency with `sbatch -d afterok:$my_id NAME.sh`
* Running on xenon:
	* Run CPU workflow only! (since no GPUs are available) with `sh cpu.sh`

Documentation:
* Slides [here](https://docs.google.com/presentation/d/1NnMhKKv6QjvK3uVa6e8yZHLHRFyKN9We8xGsiD1sIcY/edit?usp=sharing)
* Video [here](https://youtu.be/R6RBIBuJDGI)