import joblib
import contextlib
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# `rich` console used throughout the codebase
console = Console()


# `rich` progress bar used throughout the codebase
def get_progress(**kwargs):
    return Progress(
        SpinnerColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        **kwargs,
    )


@contextlib.contextmanager
def rich_progress_joblib(description=None, total=None, verbose=False):
    if description is None:
        description = "Processing..."

    progress = get_progress()
    if verbose:
        task_id = progress.add_task(description, total=total)

    class BatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if verbose:
                progress.update(task_id, advance=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack

    try:
        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallback
        progress.start()

        yield progress
    finally:
        progress.stop()
        joblib.parallel.BatchCompletionCallBack = old_callback
