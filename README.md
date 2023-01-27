# koopa-luigi
A functional implementation of koopa in luigi :)

Notes:
* If creating a local installation, install with `pip install -e .`
* Everything can be run via `koopa-luigi`
* GPU workflow only triggeres components and does not merge
* CPU / standard workflow will trigger everything

Usage:
* Only use slurm to run the workflow(s)
* Clone this repo to somewhere server-accessible
* Update `koopa.cfg` as usual
* Update the `slurm.sh` files:
	* Change the `CONFIG` parameter to the path to `koopa.cfg`
	* Add a mail account if required
	* Change resources if required
* Log into slurm
* Run either the GPU or CPU workflow separately depending on requirements with `sbatch NAME.sh`
	* The CPU workflow is optionally dependent on the GPU workflow
	* Trigger the GPU workflow with `my_id=$(sbatch --parsable NAME.sh)`
	* Add the CPU workflow as dependency with `sbatch -d afterok:$my_id NAME.sh`
