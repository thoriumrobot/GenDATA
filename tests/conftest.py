import os
import signal
import time
import pytest


def _timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded timeout")


@pytest.fixture(autouse=True)
def per_test_timeout():
    """Autouse fixture to enforce a per-test timeout via SIGALRM on Unix.
    Set PYTEST_TIMEOUT env var (seconds) to override (default 60s).
    """
    timeout_s = int(os.getenv('PYTEST_TIMEOUT', '60'))
    # Install handler and arm alarm
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    if timeout_s > 0:
        signal.alarm(timeout_s)
    try:
        yield
    finally:
        # Cancel alarm and restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def _append_progress(line: str):
    try:
        log_path = os.getenv('PYTEST_PROGRESS_LOG', 'pytest_progress.log')
        with open(log_path, 'a') as f:
            f.write(line + "\n")
    except Exception:
        pass


def pytest_runtest_logstart(nodeid, location):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    _append_progress(f"START {ts} {nodeid}")


def pytest_runtest_logfinish(nodeid, location):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    _append_progress(f"FINISH {ts} {nodeid}")


