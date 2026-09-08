"""py123d executors with their progress bars forced on, and a process pool that needs no shared-memory semaphores."""

from __future__ import annotations

import multiprocessing
import traceback
from concurrent.futures import Future
from multiprocessing.connection import Connection, Pipe, wait
from multiprocessing.process import BaseProcess
from typing import Any, cast

from py123d.common.execution import (
    Executor,
    ExecutorResources,
    Task,
    ThreadPoolExecutor,
)
from tqdm import tqdm
from typing_extensions import override


def _serve(connection: Connection) -> None:
    """Runs one worker: answers every (index, fn, args) message with (index, result, traceback)."""
    while True:
        try:
            index, fn, args = cast("tuple[int, Any, tuple[Any, ...]]", connection.recv())
        except EOFError:
            return
        try:
            connection.send((index, fn(*args), None))
        except Exception:
            connection.send((index, None, traceback.format_exc()))


class VerboseProcessPoolExecutor(Executor):
    """
    Forkserver workers driven over pipes, with progress bars on.

    The stdlib pool coordinates its workers through named semaphores in /dev/shm; a node
    cleaning that directory (logind's RemoveIPC, container limits) breaks the pool and
    hangs the parent at exit. Pipes travel as file descriptors, so nothing on the node can
    take them away, and a dead worker raises here instead of stalling.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """
        Starts the workers.

        Args:
            max_workers: number of worker processes; the node's logical core count when None.
        """
        super().__init__(
            ExecutorResources(
                number_of_nodes=1,
                number_of_gpus_per_node=0,
                number_of_cpus_per_node=max_workers or ExecutorResources.current_node_cpu_count(),
            ),
        )
        context = multiprocessing.get_context("forkserver")
        self._workers: list[tuple[BaseProcess, Connection]] = []
        for _ in range(self.number_of_threads):
            parent_connection, child_connection = cast("tuple[Connection[Any, Any], Connection[Any, Any]]", Pipe())
            process = context.Process(target=_serve, args=(child_connection,), daemon=True)
            process.start()
            child_connection.close()
            self._workers.append((process, parent_connection))

    @override
    def _map(
        self,
        task: Task,
        *item_lists: Any,
        verbose: bool = False,
        desc: str | None = None,
    ) -> list[Any]:
        """Inherited, see superclass."""
        del verbose
        name = desc or "ProcessPoolExecutor"
        calls = list(zip(*item_lists, strict=True))
        results: list[Any] = [None] * len(calls)
        idle = [connection for _, connection in self._workers]
        busy: list[Connection] = []
        next_index = 0
        with tqdm(total=len(calls), desc=name, leave=False) as progress:
            while next_index < len(calls) or busy:
                while next_index < len(calls) and idle:
                    connection = idle.pop()
                    connection.send((next_index, task.fn, calls[next_index]))
                    busy.append(connection)
                    next_index += 1
                for ready in cast("list[Connection]", wait(busy)):
                    try:
                        index, result, error = cast("tuple[int, Any, str | None]", ready.recv())
                    except EOFError:
                        self._terminate()
                        raise RuntimeError(f"{name}: a worker process died") from None
                    if error is not None:
                        self._terminate()
                        raise RuntimeError(f"{name} failed in a worker:\n{error}")
                    results[index] = result
                    busy.remove(ready)
                    idle.append(ready)
                    progress.update()
        return results

    @override
    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        raise NotImplementedError("VerboseProcessPoolExecutor only supports map")

    def _terminate(self) -> None:
        for process, connection in self._workers:
            connection.close()
            process.terminate()
        self._workers = []


class VerboseThreadPoolExecutor(ThreadPoolExecutor):
    """py123d's thread executor with its progress bars on; no py123d caller passes verbose."""

    @override
    def _map(
        self,
        task: Any,
        *item_lists: Any,
        verbose: bool = False,
        desc: str | None = None,
    ) -> list[Any]:
        """Inherited, see superclass."""
        del verbose
        return super()._map(task, *item_lists, verbose=True, desc=desc)
