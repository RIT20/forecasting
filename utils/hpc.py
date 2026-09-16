# src/forecast_engine/hpc.py

# Import libraries
from typing import Callable

from joblib import Parallel, delayed, parallel_backend

from tqdm import tqdm
import psutil

__author__ = "Nitesh Tripathi"


def parallelize(
    function: Callable,
    params: dict = None,
    backend: str = "loky",
    n_jobs: int = 1,
):
    """Parallelize a function over dataset groups

    Args:
        function (Callable): Function to apply to each group
        params (dict, optional): Parameters for the function
        backend (str, optional): Parallelization backend
        n_jobs (int, optional): No. of cores to use

    Returns:
        pd.DataFrame: Output dataframe
    """

    # Parallelize given function
    if backend == "serial":
        results = []
        for param in params:
            results.append(function(**param))
    else:
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            results = Parallel()(delayed(function)(**param) for param in tqdm(params))

    return results


def kill_child_processes():
    current_process = psutil.Process()
    sub_processes = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in sub_processes:
        try:
            psutil.Process(subproc).terminate()
        except:
            pass
