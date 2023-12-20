Here you can find examples of submitting independent MO training jobs as SLURM jobs.

In some cases, multiple MOs need to be trained simultaneously. For example, training for multiple grids with independent MOs. We recommend to use Dask to parallel the training jobs. Please refer to the [relevant documentation of MOTrianer](https://vegewaterdynamics.github.io/motrainer/).

It is also possible to submit independent training processes as SLURM jobs, if you work on an HPC system with SLURM.

This folder contains an example of scaling up a single grid training process: `jackknife_train_one.py`, to multi grids. The processes can be followed to perform the multi-grid training:

1. Configure training parameters in `jackknife_train_one.py`.
2. Configure SLURM settings in `train_multiple_gpi.slurm`.
3. Submit training processes as SLURM jobs by executing `submit.sh`. The following example will submit the training job of grids with ID 1 to 5, and submit 2 grids in each batch.

    ```bash
        bash submit.sh 1 5 2
    ```