"""YouTube videos as sources: a video's captions become time-stamped sections.

A video is read from its captions alone — the words a verdict can quote — and
never downloaded. yt-dlp finds the caption tracks (YouTube's player code, which
it must run to do so, runs in deno), the chosen track's json3 file is read, and
its timed lines are grouped into windows of WINDOW_SECONDS, each a section
whose time locator the claims view seeks to and the chat cites.

Which captions count (choose_track): a track written for the video in the
language it is spoken in; else YouTube's own speech recognition of that
language; else a written track in another language, English first. Never a
machine translation — and on a video YouTube has dubbed, every dubbed audio
track has a speech-recognition track of its own, all labelled "-orig" by
yt-dlp, so the spoken language is read from the audio track YouTube marks
original, and with none marked no automatic track is read at all.

Isolation, in the order a read meets it:

1. The input is a video id, validated here again; the link the reader opens
   is rebuilt from it (fetch.canonical_video_url), so nothing a user typed
   beyond the id — a smuggled fragment, a playlist — reaches yt-dlp.
2. yt-dlp runs in a bounded child (web.in_bounded_child): one wall-clock
   deadline, a CPU limit, a process group killed with it (deno included),
   and a parent-death watcher.
3. The child's environment is cut to what a program needs (no API keys, no
   proxy), and its DNS answers only www.youtube.com / youtube.com on port 443,
   and only with public addresses — the one host a read was measured to use.
4. yt-dlp may use its YouTube extractor alone, one video, no playlist, no
   cookies, no proxy, no cache on disk, no solver code from the network, and
   deno — which yt-dlp starts with no permission to read files, the network
   or the environment. A missing runtime is a failure, not a warning.
5. The caption file must come from YouTube's caption service, and is read
   under a byte cap. Its address carries the reader's IP address and a
   signature, so it is never logged, stored or quoted.
"""

import importlib.metadata
import json
import math
import os
import re
import socket
from collections.abc import Callable, MutableMapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import TypeGuard
from urllib.parse import urlsplit

import deno

from authorai.fetch import canonical_video_url, is_public_address, shown_text
from authorai.ingest import ParsedDocument, ParsedSection
from authorai.log import setup_logger
from authorai.web import (
    MAX_METADATA_CHARS,
    ExtractionTimeoutError,
    end_with_parent,
    in_bounded_child,
    report_outcome,
)

logger = setup_logger(__name__)

# Seconds of speech per section: about 1,100 characters of speech, a
# paragraph-sized piece of evidence whose start the claims view seeks to.
WINDOW_SECONDS = 75
# The most a caption file may be. YouTube's speech-recognition json3 runs 13-18
# KB a minute (measured), so this is ~16 hours of speech, far past the
# 200,000 characters (~4 hours) a source may contribute; a larger file is
# refused rather than cut, since a cut JSON file cannot be read.
CAPTION_MAX_BYTES = 16 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 15
YT_DLP_VERSION = importlib.metadata.version("yt-dlp")

# The hosts and port a read was measured to use (author_ai/PR_D_SPIKE.md).
READER_HOSTS = frozenset({"www.youtube.com", "youtube.com"})
READER_PORT = 443
# What the reader's environment keeps: what a program needs to start and to
# find its home and temporary directory — nothing that is a secret or a proxy.
_ENVIRONMENT_KEPT = frozenset({"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"})
# yt-dlp's ORIGINAL_LANG_VALUE: the language_preference it gives the audio
# formats of the track YouTube names "original".
_ORIGINAL_AUDIO = 10
_CAPTION_SERVICE = ("https", "www.youtube.com", "/api/timedtext")
_LANGUAGE_SUBTAG = re.compile(r"[A-Za-z]{2}|[A-Za-z]{4}|\d{3}")
# yt-dlp's advice to a command-line user, which is not ours to give.
_COMMAND_LINE_ADVICE = re.compile(r"\s*(?:Use --|See\s+https?://|Also see\s+https?://).*$", re.S)
_BOT_CHECK = (
    "YouTube asked to confirm the reader is not a bot (it asks this of live streams, and when "
    "it limits requests from this network) — try again later"
)
_UNFINISHED = {
    "is_live": "it is live now, and YouTube writes a stream's captions only after it ends — "
    "try again later",
    "is_upcoming": "it has not started yet — try again later, once it has been recorded",
    "post_live": "it has just ended, and YouTube is still processing the recording — "
    "try again later",
}


class VideoRefusedError(ValueError):
    """The video cannot be a source — private, removed, live, without usable
    captions — and the message says why, naming the video."""


class VideoReaderError(RuntimeError):
    """The reader failed, not the video: no JavaScript runtime, an answer
    that is not the video asked for, a download that broke. The message
    names the video and yt-dlp's version, and is passed on as it is."""


@dataclass(frozen=True)
class CaptionTrack:
    kind: str  # "manual" (written for the video) or "automatic" (speech recognition)
    key: str  # yt-dlp's key for the track, e.g. "en", "en-orig", "en-j3PyPqV-e1s"
    language: str  # its language tag, e.g. "en", "pt-BR"
    url: str  # the json3 file's address: never logged, stored or quoted


# --- which captions count -------------------------------------------------------


def choose_track(info: dict) -> CaptionTrack | None:
    """The one caption track a video is read from, or None when it has none
    that counts (see the module docstring for the order)."""
    manual = {
        key: formats
        for key, formats in (info.get("subtitles") or {}).items()
        if key != "live_chat" and _json3(formats)
    }
    # Plain automatic keys are machine translations; only "-orig" ones are speech.
    speech = {
        key: formats
        for key, formats in (info.get("automatic_captions") or {}).items()
        if key.endswith("-orig") and _json3(formats)
    }
    original = original_language(info)
    if original is not None:
        key = _written_in(manual, original)
        if key is not None:
            return _track("manual", key, manual)
        key = _spoken_in(speech, original)
        if key is not None:
            return _track("automatic", key, speech)
    if manual:
        key = next((key for key in manual if _primary(key) == "en"), next(iter(manual)))
        return _track("manual", key, manual)
    return None


def original_language(info: dict) -> str | None:
    """The language the video is spoken in: that of the audio track YouTube
    marks original, when the formats agree on one; else, on a video with one
    audio track (none marked), the language of its one speech-recognition
    track; else unknown."""
    marked = {
        fmt["language"]
        for fmt in info.get("formats") or []
        if fmt.get("language_preference") == _ORIGINAL_AUDIO and fmt.get("language")
    }
    if marked:
        return marked.pop() if len(marked) == 1 else None
    speech = [key for key in info.get("automatic_captions") or {} if key.endswith("-orig")]
    return _language_tag(speech[0]) if len(speech) == 1 else None


def _written_in(manual: dict, language: str) -> str | None:
    if language in manual:
        return language
    return next((key for key in manual if _primary(key) == _primary(language)), None)


def _spoken_in(speech: dict, language: str) -> str | None:
    if f"{language}-orig" in speech:
        return f"{language}-orig"
    same = [key for key in speech if _primary(key) == _primary(language)]
    return same[0] if len(same) == 1 else None


def _track(kind: str, key: str, tracks: dict) -> CaptionTrack:
    return CaptionTrack(kind=kind, key=key, language=_language_tag(key), url=_json3(tracks[key]))


def _json3(formats) -> str | None:
    for fmt in formats or []:
        if isinstance(fmt, dict) and fmt.get("ext") == "json3" and isinstance(fmt.get("url"), str):
            return fmt["url"]
    return None


def _language_tag(key: str) -> str:
    """A track key's language tag: "en-orig" -> "en", "pt-BR-orig" -> "pt-BR",
    "en-j3PyPqV-e1s" (a written track's id appended) -> "en". "orig" is taken
    off first: four letters, it would otherwise pass for a script subtag."""
    first, *rest = key.removesuffix("-orig").split("-")
    parts = [first]
    for part in rest:
        if not _LANGUAGE_SUBTAG.fullmatch(part):
            break
        parts.append(part)
    return "-".join(parts)


def _primary(tag: str) -> str:
    return tag.split("-", 1)[0].lower()


# --- captions into time-stamped sections -------------------------------------------


def transcript_windows(
    captions: object, *, window_seconds: float = WINDOW_SECONDS
) -> list[ParsedSection]:
    """A json3 caption file as sections of `window_seconds` each, titled by
    the span of speech they hold ("1:15–2:29"), with that span as their time
    locator.

    Lines are read in start order; lines displayed together overlap in time
    but never repeat words. Each line's segments are joined as written (the
    speech-recognition segments carry their own spaces) and whitespace is
    collapsed, so a line break — including YouTube's line-only events — is a
    space and consecutive lines never weld into one word. Events without text
    (display windows, blank lines) are skipped. Raises ValueError for a file
    of another shape, and VideoRefusedError when no line has any text.
    """
    events = captions.get("events") if isinstance(captions, dict) else None
    if not isinstance(events, list):
        raise ValueError("the caption file is not in YouTube's json3 shape (no list of events)")
    lines: list[tuple[float, float, str]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("the caption file is not in YouTube's json3 shape (an event)")
        segments = event.get("segs")
        if segments is None:
            continue
        start, duration = event.get("tStartMs"), event.get("dDurationMs", 0)
        if not isinstance(segments, list) or not _milliseconds(start):
            raise ValueError("the caption file is not in YouTube's json3 shape (an event's timing)")
        text = " ".join("".join(_segment_text(segment) for segment in segments).split())
        if text:
            end = start + (duration if _milliseconds(duration) else 0)
            lines.append((start / 1000, end / 1000, text))
    if not lines:
        raise VideoRefusedError("its captions have no text")
    lines.sort(key=lambda line: line[0])
    windows: dict[int, list[tuple[float, float, str]]] = {}
    for line in lines:
        windows.setdefault(int(line[0] // window_seconds), []).append(line)
    sections = []
    for index in sorted(windows):
        window = windows[index]
        start, end = window[0][0], max(line[1] for line in window)
        sections.append(
            ParsedSection(
                title=f"{clock(start)}–{clock(end)}",
                page=None,
                text=" ".join(line[2] for line in window),
                start_seconds=start,
                end_seconds=end,
            )
        )
    return sections


def _milliseconds(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _segment_text(segment: object) -> str:
    if not isinstance(segment, dict):
        raise ValueError("the caption file is not in YouTube's json3 shape (a segment)")
    text = segment.get("utf8", "")
    if not isinstance(text, str):
        raise ValueError("the caption file is not in YouTube's json3 shape (a segment's text)")
    return text


def clock(seconds: float) -> str:
    """A moment in a video as a player shows it: 0:59, 12:05, 1:01:15."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


# --- what the video says about itself ---------------------------------------------


def video_provenance(info: dict, track: CaptionTrack) -> dict:
    """What a video declares about itself, in the fields a page's markup
    fills (web.PageMetadata), plus what only a video has. The channel is the
    publisher and the upload date the publication date; a channel is never a
    personal author. `channel_verified` is YouTube's own badge — True only when
    yt-dlp says so (it reports None, not False, without one) — and decides
    whether the channel's name may earn a publisher's authority. Embeddable
    only when YouTube lets anyone play it on another site."""
    age_limit = info.get("age_limit")
    duration = info.get("duration")
    return {
        "title": _cut(info.get("title")),
        "authors": [],
        "publisher": _cut(info.get("channel")) or _cut(info.get("uploader")),
        "publication_date": _upload_date(info.get("upload_date")),
        "doi": None,
        "scholarly": False,
        "video": {
            "id": info.get("id"),
            "channel_id": _cut(info.get("channel_id")),
            "channel_verified": info.get("channel_is_verified") is True,
            "duration_seconds": duration if _milliseconds(duration) else None,
            "embeddable": info.get("playable_in_embed") is True
            and isinstance(age_limit, int)
            and age_limit < 18,
            "captions": {"kind": track.kind, "language": track.language},
        },
    }


def _cut(value: object) -> str | None:
    return value[:MAX_METADATA_CHARS] if isinstance(value, str) and value else None


def _upload_date(value: object) -> str | None:
    """yt-dlp's YYYYMMDD as an ISO date, or None when it is not a real date."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        return None
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:])).isoformat()
    except ValueError:
        return None


# --- refusals ----------------------------------------------------------------------


def reason_from_ytdlp(message: str, video_id: str) -> str:
    """YouTube's reason, from a yt-dlp error message: its "ERROR: [youtube]
    <id>:" prefix and its advice to command-line users removed, and bounded
    (fetch.shown_text) — the server decides how long it is, and it is stored
    with the run. A bot check is explained, since its own words blame nobody."""
    text = re.sub(
        rf"^(?:ERROR:\s*)?(?:\[youtube\]\s*)?(?:{re.escape(video_id)}:\s*)?", "", message.strip()
    )
    if "not a bot" in text.lower():
        return _BOT_CHECK
    return shown_text(_COMMAND_LINE_ADVICE.sub("", text).strip())


def _cannot(url: str, reason: str) -> str:
    return f"The YouTube video {url} cannot be read: {reason}"


def _could_not(url: str, reason: str) -> str:
    return f"The YouTube video {url} could not be read: {reason} (yt-dlp {YT_DLP_VERSION})"


# --- the reader ------------------------------------------------------------------


def video_options(logger) -> dict:
    """yt-dlp's options for reading one video's captions: see the module
    docstring, item 4. `logger` receives yt-dlp's messages."""
    return {
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "logger": logger,
        "skip_download": True,
        "noplaylist": True,
        "allowed_extractors": ["youtube"],
        # Stream manifests are for downloading; skipping them saves two requests.
        "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
        "proxy": "",
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "retries": 1,
        "extractor_retries": 1,
        "cachedir": False,
        "js_runtimes": {"deno": {"path": _deno_path()}},
        "remote_components": set(),
    }


def _deno_path() -> str:
    try:
        return deno.find_deno_bin()
    except FileNotFoundError as exc:
        raise VideoReaderError(
            "the video reader's JavaScript runtime (deno) is not installed — reinstall the "
            "backend's pinned dependencies (pip install -e .)"
        ) from exc


class _YtDlpLog:
    """yt-dlp's logger: warnings kept (one of them is a failure) and logged,
    bounded; its progress chatter dropped."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning("yt-dlp: %s", shown_text(message))

    def error(self, message: str) -> None:
        logger.warning("yt-dlp: %s", shown_text(message))


def read_video(video_id: str, *, ydl_class: Callable | None = None) -> tuple[ParsedDocument, dict]:
    """Read one video's captions into a document and what the video declares
    about itself. Runs in the reader process (read_video_bounded); `ydl_class`
    replaces yt_dlp.YoutubeDL in tests.

    Raises VideoRefusedError when the video cannot be a source (YouTube's
    reason, no usable captions, live or unfinished, captions too large),
    ValueError for a caption file that cannot be read, and RuntimeError for a
    failure of the reader itself (no JavaScript runtime, an unexpected
    answer) — every message naming the video."""
    url = canonical_video_url(video_id)
    from yt_dlp.utils import DownloadError  # the reader process's import, never the server's

    if ydl_class is None:
        from yt_dlp import YoutubeDL as ydl_class
    log = _YtDlpLog()
    try:
        options = video_options(log)
    except VideoReaderError as exc:
        raise VideoReaderError(_could_not(url, str(exc))) from None
    with ydl_class(options) as ydl:
        try:
            info = ydl.extract_info(url, download=False, process=False)
        except DownloadError as exc:
            raise _extraction_failure(exc, video_id, url) from None
        if any("No supported JavaScript runtime" in warning for warning in log.warnings):
            raise VideoReaderError(
                _could_not(url, "yt-dlp could not use its JavaScript runtime (deno)")
            )
        if info.get("_type") not in (None, "video") or info.get("id") != video_id:
            raise VideoReaderError(
                _could_not(url, "YouTube answered with something that is not the video asked for")
            )
        if info.get("live_status") in _UNFINISHED:
            raise VideoRefusedError(_cannot(url, _UNFINISHED[info["live_status"]]))
        track = choose_track(info)
        if track is None:
            if any("PO token" in warning for warning in log.warnings):
                raise VideoRefusedError(
                    _cannot(
                        url,
                        "YouTube withheld this video's captions from the reader (it asked for "
                        "a proof-of-origin token) — try again later",
                    )
                )
            raise VideoRefusedError(
                _cannot(
                    url,
                    "it has no usable captions (neither captions written for it nor YouTube's "
                    "automatic captions of the language it is spoken in)",
                )
            )
        raw = _read_captions(ydl, track, url)
    try:
        sections = transcript_windows(json.loads(raw))
    except VideoRefusedError as exc:
        raise VideoRefusedError(_cannot(url, str(exc))) from None
    except ValueError as exc:  # json's decoding errors included
        reason = (
            str(exc) if exc.args and "caption file" in str(exc) else "the caption file is not JSON"
        )
        raise ValueError(_cannot(url, reason)) from None
    document = ParsedDocument(
        title=_cut(info.get("title")), sections=sections, tables=[], figures=[]
    )
    return document, video_provenance(info, track)


def _extraction_failure(exc: Exception, video_id: str, url: str) -> Exception:
    """yt-dlp's failure as a refusal when YouTube refused (yt-dlp marks those
    `expected`), else as a failure of the reader."""
    cause = exc.exc_info[1] if getattr(exc, "exc_info", None) else None
    reason = reason_from_ytdlp(str(exc), video_id)
    if getattr(cause, "expected", False) or reason == _BOT_CHECK:
        return VideoRefusedError(_cannot(url, reason))
    return VideoReaderError(_could_not(url, reason))


def _read_captions(ydl, track: CaptionTrack, url: str) -> bytes:
    parts = urlsplit(track.url)
    if (parts.scheme, parts.hostname, parts.path) != _CAPTION_SERVICE:
        raise VideoReaderError(
            _could_not(url, "its caption track is not served by YouTube's caption service")
        )
    try:
        with ydl.urlopen(track.url) as response:
            data = response.read(CAPTION_MAX_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 - yt-dlp's network errors; its text may hold the address
        status = getattr(exc, "status", None)
        detail = f"{type(exc).__name__}" + (f", HTTP {status}" if status else "")
        raise RuntimeError(
            _could_not(url, f"its captions could not be downloaded ({detail})")
        ) from None
    if len(data) > CAPTION_MAX_BYTES:
        raise VideoRefusedError(
            _cannot(url, f"its captions are larger than {CAPTION_MAX_BYTES:,} bytes")
        )
    return data


# --- the reader process --------------------------------------------------------------


def scrub_environment(environ: MutableMapping[str, str]) -> None:
    """Cut the reader's environment to _ENVIRONMENT_KEPT, and turn yt-dlp's
    plugins off. yt-dlp hands deno a copy of it."""
    for name in list(environ):
        if name not in _ENVIRONMENT_KEPT:
            del environ[name]
    environ["YTDLP_NO_PLUGINS"] = "1"


def guarded_getaddrinfo(real: Callable) -> Callable:
    """`real` (socket.getaddrinfo) answering only READER_HOSTS on READER_PORT,
    and only with public addresses: refused before any lookup otherwise, and
    after it when any answer is private or reserved (one such answer blocks
    the host, as in fetch.py)."""

    def guarded(host, port, *args, **kwargs):
        name = host.decode("ascii", "replace") if isinstance(host, bytes) else str(host or "")
        name = name.lower().rstrip(".")
        if name not in READER_HOSTS or str(port) != str(READER_PORT):
            raise socket.gaierror(
                socket.EAI_NONAME,
                f"the video reader may reach only YouTube, not {shown_text(name)!r} port {port}",
            )
        answers = real(name, port, *args, **kwargs)
        if not all(is_public_address(answer[4][0]) for answer in answers):
            raise socket.gaierror(
                socket.EAI_NONAME, f"{name!r} resolves to a private or reserved network address"
            )
        return answers

    return guarded


def prepare_reader_process() -> None:
    """The reader process's own isolation, before yt-dlp is imported."""
    scrub_environment(os.environ)
    socket.getaddrinfo = guarded_getaddrinfo(socket.getaddrinfo)


def _read_in_child(sender, _payload_path, video_id: str, cpu_seconds: int) -> None:
    end_with_parent(cpu_seconds)
    prepare_reader_process()

    def read():
        document, provenance = read_video(video_id)
        return document.title, [asdict(section) for section in document.sections], provenance

    report_outcome(sender, read, _failure_kind)


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, VideoRefusedError):
        return "refused"
    if isinstance(exc, VideoReaderError):
        return "failed"
    return "value" if isinstance(exc, ValueError) else "other"


def read_video_bounded(video_id: str, *, timeout: float) -> tuple[ParsedDocument, dict]:
    """read_video in a bounded child process (web.in_bounded_child), stopped
    after `timeout` seconds of wall clock. The child's refusal and ValueError
    come back as the same type with the same message, and so does its own
    failure (VideoReaderError, which already names the video); anything else
    is a RuntimeError naming the video, with the child's traceback in the log."""
    url = canonical_video_url(video_id)
    try:
        kind, payload = in_bounded_child(
            _read_in_child, (video_id,), payload_path=None, url=url, timeout=timeout
        )
    except ExtractionTimeoutError:
        raise ExtractionTimeoutError(
            f"The YouTube video {url} took longer than {timeout:g} seconds to read — "
            "YouTube may be slow to answer; try again later"
        ) from None
    if kind == "result":
        title, sections, provenance = payload
        document = ParsedDocument(
            title=title,
            sections=[ParsedSection(**section) for section in sections],
            tables=[],
            figures=[],
        )
        return document, provenance
    name, message, child_traceback = payload
    if kind == "refused":
        raise VideoRefusedError(message)
    logger.warning("reading %s failed in the reader process:\n%s", url, child_traceback)
    if kind == "failed":
        raise VideoReaderError(message)
    if kind == "value":
        raise ValueError(message)
    raise RuntimeError(f"{url} could not be read: {name}: {message}")
