"""The bounded reader (web.extract_web_bounded): extract_web in a spawned child
under a wall-clock budget, on the single worker thread.

What is at stake is the worker thread itself: a page that holds it, a child
that wedges its start, or a reader that outlives the server all cost the same
thing. Every test that kills, orphans or stalls a process is bounded by
construction (a joined daemon thread, a subprocess timeout, a finally that
kills) so a regression fails instead of hanging the suite.
"""

import multiprocessing
import multiprocessing.spawn
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from multiprocessing import resource_tracker

import pytest

import authorai.web as web_mod
from authorai.web import ExtractionTimeoutError, ThinPageError, extract_web, extract_web_bounded
from tests.test_web import (
    BACKEND,
    DIARIO_URL,
    FIXTURES,
    REPORT_URL,
    SPA_URL,
    _diario,
    _page,
    _prose,
    _subprocess_env,
)


def _slow_page(tag: str, count: int = 30_000) -> str:
    """A page inside the fetch cap made of one inline tag trafilatura strips
    ITSELF (<abbr>, <cite>, <mark>, <small>, ...): its cleaning leaves a run of
    adjacent text nodes and then runs XPaths over it, quadratic in the count.
    _prepare_tree never touches these tags, so the linear rewrite cannot help;
    30,000 of them (0.4 MB) take about 6 s here, an hour at the cap."""
    return (
        "<!DOCTYPE html><html><head><title>Slow</title></head><body><main><article>"
        f"<h1>Slow</h1><p>{_prose(400)}</p><p>"
        + f"<{tag}>x</{tag}>" * count
        + "</p></article></main></body></html>"
    )


def _running(pid: int) -> bool:
    """Whether a process exists and is not a zombie (an orphan is reaped by
    launchd/init; a child of ours stays a zombie until joined)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return state.returncode == 0 and not state.stdout.strip().startswith("Z")


def _signal_the_reader(before: set, signum: int, box: dict) -> None:
    """From a helper thread: signal the first reader that appears beyond
    `before`. The child is registered the moment start() returns, so the
    signal lands during its start-up or its read, never after."""
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and "pid" not in box:
        try:
            new = set(multiprocessing.active_children()) - before
        except RuntimeError:  # the set changed while start() added the child
            continue
        for child in new:
            os.kill(child.pid, signum)
            box["pid"] = child.pid
        time.sleep(0.005)


@pytest.fixture()
def reader_that_exits_at_once():
    """Every spawned child becomes `false`: it exits before reading anything the
    parent sent it — a reader killed during start-up, or one whose import of
    authorai.web fails because a module on disk was edited under a running
    server (the child imports from disk; the parent runs what it loaded)."""
    resource_tracker.ensure_running()  # launched with the same executable: keep it real
    original = multiprocessing.spawn.get_executable()
    multiprocessing.spawn.set_executable(shutil.which("false"))
    try:
        yield
    finally:
        multiprocessing.spawn.set_executable(original)


# --- the contract: same result, same errors, stopped at the deadline ----------


def test_bounded_extraction_returns_exactly_what_extract_web_returns():
    page = _page("report_jsonld_graph.html")
    bounded = extract_web_bounded(page, url=REPORT_URL, timeout=30)
    assert bounded == extract_web(page, url=REPORT_URL)
    raw = (FIXTURES / "diario_agua_es.html").read_bytes()  # bytes cross the process line too
    assert extract_web_bounded(raw, url=DIARIO_URL, timeout=30) == extract_web(raw, url=DIARIO_URL)


def test_bounded_extraction_reraises_thin_page_and_decoding_errors_as_the_same_type():
    with pytest.raises(ThinPageError) as bounded:
        extract_web_bounded(_page("spa_shell.html"), url=SPA_URL, timeout=30)
    with pytest.raises(ThinPageError) as direct:
        extract_web(_page("spa_shell.html"), url=SPA_URL)
    assert type(bounded.value) is ThinPageError
    assert str(bounded.value) == str(direct.value)

    undeclared = _diario("")
    with pytest.raises(ValueError) as decoding:
        extract_web_bounded(undeclared.encode("cp1252"), url=DIARIO_URL, timeout=30)
    assert type(decoding.value) is ValueError
    assert DIARIO_URL in str(decoding.value)


def test_a_page_that_takes_too_long_to_read_is_stopped_at_the_deadline_and_leaves_no_child(
    monkeypatch,
):
    # What the bound must prove: the deadline fires while the reader is still
    # at work, and the reader is then stopped, not awaited. A wall-clock
    # ceiling that includes a fresh interpreter's start-up flakes under load,
    # so the ceiling here is generous and the proof is in the stop itself.
    stops, real_stop = [], web_mod._stop

    def timed_stop(child):
        alive, pid, began = child.is_alive(), child.pid, time.perf_counter()
        result = real_stop(child)
        stops.append((alive, pid, time.perf_counter() - began))
        return result

    monkeypatch.setattr(web_mod, "_stop", timed_stop)
    children = set(multiprocessing.active_children())
    started = time.perf_counter()
    with pytest.raises(ExtractionTimeoutError) as excinfo:
        extract_web_bounded(_slow_page("abbr", 60_000), url="https://example.org/slow", timeout=0.5)
    assert time.perf_counter() - started < 0.5 + 15.0  # the page alone takes 25 s and more
    assert str(excinfo.value) == (
        "https://example.org/slow took longer than 0.5 seconds to read "
        "(the page is too large or complex)"
    )
    assert isinstance(excinfo.value, RuntimeError)
    [(alive, pid, seconds)] = stops
    assert alive  # the deadline fired with the reader still at work
    assert seconds < 5.0  # terminated, not awaited
    assert not _running(pid)
    assert set(multiprocessing.active_children()) == children


@pytest.mark.parametrize("tag", ["abbr", "cite"])
def test_inline_tags_trafilatura_strips_itself_are_stopped_by_the_time_bound(tag):
    # The shape that bypasses the linear tree edits: only the wall-clock bound
    # stands between it and the single worker thread.
    with pytest.raises(ExtractionTimeoutError, match="too large or complex"):
        extract_web_bounded(_slow_page(tag), url=f"https://example.org/{tag}", timeout=0.5)


def test_the_reader_runs_in_a_fresh_interpreter_not_a_fork_of_the_caller(monkeypatch):
    # The caller is the worker thread of a multi-threaded server: a forked
    # child would inherit its memory, locks other threads hold included.
    # Poison the caller's copy of the module; a spawned reader never sees it.
    monkeypatch.setattr(web_mod, "MIN_BODY_CHARS", 10**9)
    document, _ = extract_web_bounded(_page("report_jsonld_graph.html"), url=REPORT_URL, timeout=30)
    assert document.sections


# --- the deadline covers the child's start-up --------------------------------


def test_a_reader_that_dies_before_reading_its_page_fails_at_once_whatever_the_page_size(
    reader_that_exits_at_once, tmp_path, monkeypatch
):
    # The spawn launcher writes the Process arguments to the child over a pipe
    # while still holding the child's end of it, so arguments beyond the pipe
    # buffer (64 KiB) would park child.start() forever once the child is gone —
    # before any deadline is in force. The page must not travel that way.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    page = (
        "<!DOCTYPE html><html><head><title>Big</title></head><body><main><article>"
        f"<h1>Big</h1><p>{_prose(300_000)}</p></article></main></body></html>"
    )
    assert len(page.encode()) > 65536
    children = set(multiprocessing.active_children())
    failures: list[Exception] = []

    def read():
        try:
            extract_web_bounded(page, url="https://example.org/big", timeout=5)
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(5 + 15)
    assert not thread.is_alive(), "extract_web_bounded blocked inside child.start()"
    assert len(failures) == 1 and type(failures[0]) is web_mod.ReaderExitedError
    assert str(failures[0]) == (
        "https://example.org/big could not be read: the reader process exited without a result"
    )
    assert list(tmp_path.glob("authorai-page-*")) == []  # the page file went with the child
    assert set(multiprocessing.active_children()) == children


def test_the_time_bound_holds_while_the_reader_is_still_starting(tmp_path, monkeypatch):
    # A loaded host makes the spawned reader slow to start; simulate it
    # deterministically: every child sleeps 8 s before it can read anything.
    resource_tracker.ensure_running()  # before the sleep applies to new interpreters
    (tmp_path / "sitecustomize.py").write_text("import time\ntime.sleep(8)\n")
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) if not existing else f"{tmp_path}{os.pathsep}{existing}"
    )
    children = set(multiprocessing.active_children())
    started = time.perf_counter()
    with pytest.raises(ExtractionTimeoutError):
        extract_web_bounded(_slow_page("abbr"), url="https://example.org/slow", timeout=0.5)
    assert time.perf_counter() - started < 0.5 + 4.0  # never the child's 8 s sleep
    assert set(multiprocessing.active_children()) == children


# --- a reader that dies, and one that fails unexpectedly ---------------------


def test_a_reader_killed_mid_read_fails_naming_the_link_and_logs_the_signal(web_log):
    # An out-of-memory kill, from the parent's side: the pipe closes with no
    # result. That must be the documented RuntimeError at once (the child holds
    # the only writing end), not a TypeError and not the deadline.
    before = set(multiprocessing.active_children())
    killed: dict = {}
    killer = threading.Thread(
        target=_signal_the_reader, args=(before, signal.SIGKILL, killed), daemon=True
    )
    killer.start()
    with pytest.raises(RuntimeError) as excinfo:
        extract_web_bounded(_slow_page("abbr", 60_000), url="https://example.org/oom", timeout=20)
    killer.join(5)
    assert killed
    assert type(excinfo.value) is web_mod.ReaderExitedError  # not the deadline's error
    assert excinfo.value.exitcode == -signal.SIGKILL  # the child's own status, for the caller
    assert str(excinfo.value) == (
        "https://example.org/oom could not be read: the reader process exited without a result"
    )
    assert not _running(killed["pid"])
    assert set(multiprocessing.active_children()) == before
    assert "https://example.org/oom" in web_log.text
    assert "killed by signal SIGKILL" in web_log.text


def test_any_other_failure_in_the_reader_names_the_link_and_logs_its_traceback(web_log):
    # A failure that is neither thin-page nor decoding, raised by the real
    # reader: the run's error names the link and the type, and the frame it
    # came from goes to the log — the child's traceback, not the re-raise here.
    children = set(multiprocessing.active_children())
    with pytest.raises(RuntimeError) as excinfo:
        extract_web_bounded(_page("report_jsonld_graph.html"), url=12345, timeout=30)
    assert type(excinfo.value) is RuntimeError
    assert str(excinfo.value).startswith("12345 could not be read: AttributeError: ")
    assert set(multiprocessing.active_children()) == children
    assert "12345" in web_log.text
    assert "Traceback (most recent call last)" in web_log.text
    assert "in extract_web" in web_log.text  # the child's own frames


def _reader_dies_mid_report(sender, *_args):
    """A reader killed while writing its result (out of memory, jetsam): the
    frame header promises more bytes than ever arrive."""
    fd = sender.fileno()
    os.write(fd, struct.pack("!i", 1_000_000))
    os.write(fd, b"\x80\x04partial")
    os._exit(0)


def test_a_reader_killed_while_reporting_fails_naming_the_link(monkeypatch):
    monkeypatch.setattr(web_mod, "_extract_in_child", _reader_dies_mid_report)
    children = set(multiprocessing.active_children())
    with pytest.raises(RuntimeError) as excinfo:
        extract_web_bounded(_page("report_jsonld_graph.html"), url=REPORT_URL, timeout=30)
    assert str(excinfo.value) == (
        f"{REPORT_URL} could not be read: the reader process exited without a result"
    )
    assert set(multiprocessing.active_children()) == children


def _reader_reports_too_much(sender, *_args):
    """A reader whose result is larger than the parent will receive."""
    web_mod.report_outcome(sender, lambda: "x" * (2 * 2**20), lambda exc: "other")


def test_a_result_larger_than_the_parents_bound_is_refused_unread(monkeypatch, web_log):
    """The parent receives a result only up to RESULT_MAX_BYTES: a longer
    frame is refused on its length header before a byte of its body is
    read, as a bound of the reader's (ResultTooLargeError) — not the
    'exited without a result' a reader that dies mid-report is — naming
    the link, with the child reaped and nothing left behind."""
    monkeypatch.setattr(web_mod, "RESULT_MAX_BYTES", 2**20)
    monkeypatch.setattr(web_mod, "_extract_in_child", _reader_reports_too_much)
    children = set(multiprocessing.active_children())
    with pytest.raises(web_mod.ResultTooLargeError) as excinfo:
        extract_web_bounded(_page("report_jsonld_graph.html"), url=REPORT_URL, timeout=30)
    assert not isinstance(excinfo.value, web_mod.ReaderExitedError)
    assert str(excinfo.value) == (
        f"{REPORT_URL} could not be read: the reader reported a result larger than {2**20} bytes"
    )
    assert f"reported a result larger than {2**20} bytes" in web_log.text
    assert "exited without a result" not in web_log.text
    assert set(multiprocessing.active_children()) == children


def test_a_page_that_cannot_be_handed_to_the_reader_fails_naming_the_link(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
    children = set(multiprocessing.active_children())
    with pytest.raises(RuntimeError) as excinfo:
        extract_web_bounded(_page("report_jsonld_graph.html"), url=REPORT_URL, timeout=30)
    assert str(excinfo.value).startswith(f"{REPORT_URL} could not be read: FileNotFoundError: ")
    assert set(multiprocessing.active_children()) == children


def test_the_page_file_is_removed_on_every_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    extract_web_bounded(_page("report_jsonld_graph.html"), url=REPORT_URL, timeout=30)
    assert list(tmp_path.glob("authorai-page-*")) == []
    with pytest.raises(ThinPageError):
        extract_web_bounded(_page("spa_shell.html"), url=SPA_URL, timeout=30)
    assert list(tmp_path.glob("authorai-page-*")) == []
    with pytest.raises(ExtractionTimeoutError):
        extract_web_bounded(_slow_page("abbr"), url="https://example.org/slow", timeout=0.5)
    assert list(tmp_path.glob("authorai-page-*")) == []


# --- the reader's life is tied to the server's -------------------------------


def test_a_ctrl_c_reaching_the_reader_never_loses_the_page():
    # A terminal Ctrl-C signals the server's whole process group, the reader
    # included. A reader that dies of it reports nothing: the run is recorded
    # FAILED blaming the link instead of staying RUNNING for startup recovery.
    # SIGINT is sent the moment the reader exists, while it is still starting.
    page = _page("report_jsonld_graph.html")
    before = set(multiprocessing.active_children())
    signalled: dict = {}
    outcome: dict = {}

    def read():
        try:
            outcome["value"] = extract_web_bounded(page, url=REPORT_URL, timeout=60)
        except Exception as exc:
            outcome["value"] = exc

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    _signal_the_reader(before, signal.SIGINT, signalled)
    reader.join(60)
    assert signalled
    assert not reader.is_alive()
    assert outcome["value"] == extract_web(page, url=REPORT_URL)


_DYING_SERVER = textwrap.dedent(
    """
    import glob, multiprocessing, os, signal, sys, tempfile, threading, time
    import authorai.web as web
    print(web.__file__, flush=True)
    page = open(sys.argv[1], encoding="utf-8").read()
    threading.Thread(
        target=web.extract_web_bounded, args=(page,),
        kwargs={"url": "https://example.org/slow", "timeout": 60.0}, daemon=True,
    ).start()
    deadline = time.monotonic() + 30
    reader = None
    while reader is None and time.monotonic() < deadline:
        try:
            reader = next(iter(multiprocessing.active_children()), None)
        except RuntimeError:
            pass
        time.sleep(0.01)
    if reader is None:
        sys.exit(2)
    # Past its start-up, into the page: the reader deletes its page file once it
    # has read it. Waited for at most 15 s, well inside the page's reading time.
    pages = os.path.join(glob.escape(tempfile.gettempdir()), "authorai-page-*")
    deadline = time.monotonic() + 15
    while glob.glob(pages) and time.monotonic() < deadline:
        time.sleep(0.01)
    print(reader.pid, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    """
)


def test_a_reader_whose_server_dies_stops_reading(tmp_path):
    # A server killed outright (SIGTERM's default action after uvicorn's
    # graceful stop, SIGKILL, a crash) runs no exit hook: nothing stops the
    # reader, which reads on for as long as the page takes, and startup
    # recovery starts another beside it. Nor does the server remove the page
    # file: the reader has already done so, or every such kill would leave one
    # (up to the 10 MB fetch cap) in the temporary directory.
    page = tmp_path / "slow.html"
    page.write_text(_slow_page("abbr", 60_000), encoding="utf-8")  # 25 s and more, unbounded
    server_tmp = tmp_path / "server-tmp"
    server_tmp.mkdir()
    errors = tmp_path / "stderr.txt"
    with errors.open("w") as stderr:  # a file, not a pipe: an orphan would hold a pipe open
        server = subprocess.Popen(
            [sys.executable, "-c", _DYING_SERVER, str(page)],
            cwd=BACKEND,
            env={**_subprocess_env(), "TMPDIR": str(server_tmp)},
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
    module = server.stdout.readline().strip()
    pid_line = server.stdout.readline().strip()
    server.stdout.close()
    assert server.wait(timeout=60) == -signal.SIGKILL, errors.read_text()
    assert module == web_mod.__file__, errors.read_text()
    reader = int(pid_line)
    try:
        deadline = time.monotonic() + 10
        while _running(reader) and time.monotonic() < deadline:
            time.sleep(0.05)
        orphaned = _running(reader)
    finally:
        if _running(reader):
            os.kill(reader, signal.SIGKILL)
    assert not orphaned, "the reader kept running after its server died"
    assert list(server_tmp.glob("authorai-page-*")) == []


def test_a_reader_carries_a_cpu_limit_that_ends_it_without_its_parent():
    # The backstop for a reader the watcher cannot reach (a long C call holding
    # the GIL): the kernel ends it after its budget, no parent and no GIL needed.
    script = "import authorai.web as web\nweb._limit_cpu(2)\nwhile True:\n    pass\n"
    spinner = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert spinner.returncode == -signal.SIGXCPU, spinner.stderr


def test_the_cpu_limit_never_loosens_a_stricter_inherited_one():
    script = textwrap.dedent(
        """
        import resource
        import authorai.web as web
        infinity = resource.RLIM_INFINITY
        resource.setrlimit(resource.RLIMIT_CPU, (30, infinity))
        web._limit_cpu(65)
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        print(soft, hard == infinity)
        resource.setrlimit(resource.RLIMIT_CPU, (30, 40))
        web._limit_cpu(65)
        print(resource.getrlimit(resource.RLIMIT_CPU))
        web._limit_cpu(35)
        print(resource.getrlimit(resource.RLIMIT_CPU))
        """
    )
    probe = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == ["30 True", "(30, 40)", "(30, 40)"]


def test_the_cpu_budget_follows_the_wall_clock_budget_with_a_margin():
    # Single-threaded, a reader's CPU time never exceeds its wall time, so a
    # reader with a live parent always meets the deadline first.
    assert web_mod._orphan_cpu_seconds(60) == 65
    assert web_mod._orphan_cpu_seconds(0.5) == 6
