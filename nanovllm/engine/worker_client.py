from multiprocessing.connection import wait
from time import perf_counter


class RunnerClient:

    def __init__(self, processes, connections, timeout):
        self.processes = processes
        self.connections = connections
        self.timeout = timeout

    def receive(self, expected, all_results=False):
        pending = set(range(len(self.connections)))
        results = [None] * len(self.connections)
        deadline = perf_counter() + self.timeout
        while pending:
            remaining = deadline - perf_counter()
            ready = wait([self.connections[i] for i in pending] +
                         [self.processes[i].sentinel for i in pending], timeout=max(remaining, 0))
            if not ready:
                raise TimeoutError(f"Model workers did not finish {expected} within {self.timeout}s")
            for i in list(pending):
                if self.connections[i] in ready:
                    try:
                        kind, payload = self.connections[i].recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f"Model worker {i} disconnected") from exc
                    if kind == "error":
                        raise RuntimeError(f"Model worker {i} failed:\n{payload}")
                    if kind != expected:
                        raise RuntimeError(f"Model worker {i}: expected {expected}, received {kind}")
                    results[i] = payload
                    pending.remove(i)
                elif self.processes[i].sentinel in ready:
                    raise RuntimeError(f"Model worker {i} exited unexpectedly "
                                       f"(exit code {self.processes[i].exitcode})")
        return results if all_results else results[0]

    def call(self, method_name, *args):
        for i, connection in enumerate(self.connections):
            try:
                connection.send((method_name, args))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(f"Could not send {method_name} to model worker {i}") from exc
        return self.receive("ok")

    def call_all(self, method_name, *args):
        """Collect rank-local results, in model worker rank order."""
        for i, connection in enumerate(self.connections):
            try:
                connection.send((method_name, args))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(f"Could not send {method_name} to model worker {i}") from exc
        return self.receive("ok", all_results=True)

    def call_each(self, method_name, args_by_rank):
        """Pass a different KV shard to each rank, waiting for every import."""
        if len(args_by_rank) != len(self.connections):
            raise ValueError("Expected one set of arguments per model worker")
        for i, (connection, args) in enumerate(zip(self.connections, args_by_rank)):
            try:
                connection.send((method_name, args))
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(f"Could not send {method_name} to model worker {i}") from exc
        return self.receive("ok", all_results=True)
