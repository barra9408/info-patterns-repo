import os
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

def blas_single_thread_context():
    """
    Limit BLAS/OpenMP-backed numerical kernels to one thread inside each worker.

    This is useful when using ProcessPoolExecutor, because otherwise each
    process may internally spawn several BLAS threads, causing CPU
    oversubscription and slower parallel execution.
    """

    if threadpool_limits is None:
        return nullcontext()

    return threadpool_limits(limits=1)

def resolve_parallel_execution(parallel: bool | str, max_workers: int | None, n_tasks: int, verbose: bool) -> tuple[bool, int | None]:
    """
    Resolve serial or parallel execution.

    Parameters
    ----------
    parallel : bool | str
        Execution mode. Accepted values are False, True, and "auto".

    max_workers : int | None
        Maximum number of worker processes requested by the caller.

    n_tasks : int
        Number of independent parallel tasks.

    verbose : bool
        If True, print execution-mode messages.

    Returns
    -------
    use_parallel : bool
        Whether to use parallel execution.

    resolved_workers : int | None
        Number of worker processes if parallel execution is used.
    """

    if isinstance(parallel, str):
        parallel = parallel.lower()

    if parallel not in {False, True, "auto"}:
        raise ValueError("parallel must be one of False, True, or 'auto'.")

    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1.")

    available_cpus = os.cpu_count() or 1

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1 or None.")

    if max_workers is None:
        resolved_workers = min(available_cpus, n_tasks)
    else:
        resolved_workers = min(max_workers, available_cpus, n_tasks)

    if parallel is False:
        if verbose:
            print("Running in serial mode because parallel=False.")
        return False, None

    if resolved_workers < 2:
        if verbose:
            print(f"Not enough workers for parallel execution "
                f"({resolved_workers}). Running in serial mode.")
        return False, None

    if parallel == "auto":
        if verbose:
            print(f"Running in automatic parallel mode with "
                f"{resolved_workers} workers.")
        return True, resolved_workers

    if parallel is True:
        if verbose:
            print(f"parallel=True requested. Running in parallel mode with "
                f"{resolved_workers} workers.")
        return True, resolved_workers

    raise RuntimeError("Unreachable parallel execution state.")