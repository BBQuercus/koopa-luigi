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
* Run either the GPU or CPU workflow depending on requirements with `sbatch NAME.sh`
