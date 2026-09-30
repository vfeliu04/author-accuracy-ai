"""YouTube videos as sources (authorai.video): which caption track is read,
how its events become time-stamped sections, what a video says about itself,
and every way a video is refused.

The fixtures are real (tests/fixtures/youtube, see its README): the caption
tracks and audio formats of eight public videos as yt-dlp 2026.8.19 reported
them, and two real caption files — one written by a person, one YouTube's own
speech recognition. Nothing here touches the network; the live reader is
tests/test_youtube_integration.py.
"""

import json
import os
import socket
from pathlib import Path

import pytest

from authorai import video as videomod
from authorai.ingest import ParsedDocument, ParsedSection
from authorai.video import (
    CaptionTrack,
    VideoRefusedError,
    choose_track,
    reason_from_ytdlp,
    transcript_windows,
    video_options,
    video_provenance,
)
from tests.test_web import BACKEND, _subprocess_env

FIXTURES = Path(__file__).parent / "fixtures" / "youtube"
TRACKS = json.loads((FIXTURES / "tracks.json").read_text(encoding="utf-8"))
MANUAL_BYTES = (FIXTURES / "HBtdbaSKexU.en.json3").read_bytes()
MANUAL = json.loads(MANUAL_BYTES)
ASR = json.loads((FIXTURES / "BVVzkThVMg4.en-orig.json3").read_text(encoding="utf-8"))


def _formats(keys):
    """The formats yt-dlp lists for one caption track: json3 among others."""
    return {
        key: [
            {"ext": ext, "url": f"https://www.youtube.com/api/timedtext?v=x&lang={key}&fmt={ext}"}
            for ext in ("json3", "srv3", "vtt")
        ]
        for key in keys
    }


def _planted(*, manual=(), automatic=(), audio=()) -> dict:
    return {
        "subtitles": _formats(manual),
        "automatic_captions": _formats(automatic),
        "formats": [
            {
                "language": language,
                "language_preference": preference,
                "acodec": "opus",
                "vcodec": "none",
            }
            for language, preference in audio
        ],
    }


def _info(video_id: str) -> dict:
    """A yt-dlp info dict rebuilt from a real video's recorded shape. Its
    automatic captions also carry two auto-TRANSLATED tracks (plain keys),
    as every real one does: they must never be chosen."""
    shape = TRACKS[video_id]
    orig = shape["automatic_captions_orig"]
    return {
        **shape["fields"],
        "id": video_id,
        **_planted(
            manual=shape["subtitles"],
            automatic=[*orig, "fr", "de"] if orig else (),
            audio=shape["audio_formats"],
        ),
    }


def _chosen(info: dict) -> tuple | None:
    track = choose_track(info)
    return None if track is None else (track.kind, track.key, track.language)


# --- which captions count -----------------------------------------------------


@pytest.mark.parametrize(
    ("video_id", "expected"),
    [
        # A manual track only.
        ("HBtdbaSKexU", ("manual", "en", "en")),
        # Two manual tracks and nothing marking the original: English first.
        ("jNQXAC9IVRw", ("manual", "en", "en")),
        # The original audio is en-US; its manual track carries a track id.
        ("xAcTmDO6NTI", ("manual", "en-j3PyPqV-e1s", "en")),
        # Auto-dubbed: 18 "-orig" tracks, Arabic listed first. The original
        # audio is en-US, so the English speech-recognition track is read.
        ("BVVzkThVMg4", ("automatic", "en-orig", "en")),
        # Auto-dubbed with a manual English track: the manual one wins.
        ("_ketUOAj30k", ("manual", "en", "en")),
        ("akM4WmGJ_j8", ("manual", "en", "en")),
        # One audio track, one "-orig" track: that is the original.
        ("xVLjBeBudSs", ("automatic", "en-orig", "en")),
        # No captions at all.
        ("C1wFmXGPbUg", None),
    ],
)
def test_the_track_read_is_the_best_one_in_the_videos_own_language(video_id, expected):
    assert _chosen(_info(video_id)) == expected


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        # Original German: its own automatic track beats a manual one in
        # another language.
        (
            _planted(manual=["en", "fr"], automatic=["de-orig", "en", "fr"], audio=[("de", 10)]),
            ("automatic", "de-orig", "de"),
        ),
        # Original German, no German track of either kind: a manual track in
        # another language, English first.
        (_planted(manual=["fr", "en-GB"], audio=[("de", 10)]), ("manual", "en-GB", "en-GB")),
        (_planted(manual=["fr", "es"], audio=[("de", 10)]), ("manual", "fr", "fr")),
        # Several "-orig" tracks and nothing saying which is the original: no
        # automatic track is read, however English one of them is.
        (_planted(automatic=["ar-orig", "en-orig", "en", "ar"]), None),
        (_planted(manual=["es"], automatic=["ar-orig", "en-orig"]), ("manual", "es", "es")),
        # Two audio tracks both marked original, in different languages: unknown.
        (_planted(automatic=["de-orig", "en-orig"], audio=[("de", 10), ("en", 10)]), None),
        # The original's own automatic track by its exact tag, then by its
        # primary language when that is unique.
        (
            _planted(automatic=["pt-BR-orig", "pt-PT-orig"], audio=[("pt-PT", 10)]),
            ("automatic", "pt-PT-orig", "pt-PT"),
        ),
        (
            _planted(automatic=["pt-BR-orig", "pt-PT-orig"], audio=[("pt", 10)]),
            None,  # two Portuguese tracks, neither the original's exact tag
        ),
        # Auto-translated tracks only, and a live chat replay: nothing to read.
        (_planted(automatic=["en", "fr"]), None),
        (_planted(manual=["live_chat"]), None),
        # A track without the json3 format is skipped, not read in another format.
        (
            {
                "subtitles": {
                    "en": [{"ext": "vtt", "url": "https://www.youtube.com/api/timedtext?v=x"}]
                },
                "automatic_captions": _formats(["en-orig"]),
                "formats": [],
            },
            ("automatic", "en-orig", "en"),
        ),
    ],
)
def test_track_choice_on_planted_shapes(info, expected):
    assert _chosen(info) == expected


def test_the_chosen_track_carries_its_json3_address():
    track = choose_track(_info("HBtdbaSKexU"))
    assert isinstance(track, CaptionTrack)
    assert track.url == "https://www.youtube.com/api/timedtext?v=x&lang=en&fmt=json3"


# --- captions into time-stamped sections --------------------------------------


def _text(sections: list[ParsedSection]) -> str:
    return " ".join(section.text for section in sections)


def test_manual_captions_become_75_second_windows_of_plain_text():
    sections = transcript_windows(MANUAL)
    # 4:15 of video: windows starting at 0:00, 1:15, 2:30 and 3:45.
    assert [(s.start_seconds // 75) for s in sections] == [0, 1, 2, 3]
    assert all(s.page is None for s in sections)
    first = sections[0]
    clock = videomod.clock
    assert first.title == f"{clock(first.start_seconds)}–{clock(first.end_seconds)}"
    assert first.start_seconds == 0.01
    # Consecutive captions are separate lines, never welded into one word.
    assert first.text.startswith(
        "[ MUSIC ] The Power of Light - presented by Science@NASA Crewmembers on the "
        "International Space Station (ISS)"
    )
    assert "\n" not in _text(sections)
    assert "  " not in _text(sections)


def _words(captions: dict) -> list[str]:
    """Every word of a caption file, in file order."""
    return [
        word
        for event in captions["events"]
        for seg in event.get("segs", [])
        for word in seg.get("utf8", "").split()
    ]


def test_every_word_of_the_captions_is_kept_in_order():
    assert _text(transcript_windows(MANUAL)).split() == _words(MANUAL)


def test_speech_recognition_captions_lose_their_line_events_but_no_words():
    sections = transcript_windows(ASR)
    text = _text(sections)
    assert text.startswith(
        "As we all know, this morning we're gathered here primarily for the purpose of uh "
        "the crew being briefed by the students on the two experiments slated to fly on STS-26."
    )
    assert text.split() == _words(ASR)
    # The first event is a display window with no text; the first words are at 18.32 s.
    assert sections[0].start_seconds == 18.32
    # Displayed lines overlap in time, but a window's end is its last caption's.
    assert all(s.start_seconds < s.end_seconds for s in sections)
    assert [s.start_seconds // 75 for s in sections] == [0, 1, 2]


def _events(*events) -> dict:
    return {"wireMagic": "pb3", "events": list(events)}


def _event(start_ms, text, duration_ms=2000, **extra) -> dict:
    return {"tStartMs": start_ms, "dDurationMs": duration_ms, "segs": [{"utf8": text}], **extra}


def test_events_are_read_in_start_order_and_blank_ones_are_skipped():
    sections = transcript_windows(
        _events(
            _event(5_000, "second"),
            _event(1_000, "first"),
            {"tStartMs": 0, "dDurationMs": 60_000, "id": 1},  # a window with no text
            _event(7_000, "   "),
            {"tStartMs": 8_000, "dDurationMs": 100, "aAppend": 1, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 9_000, "segs": [{}]},
            _event(10_000, "third line"),
        )
    )
    [only] = sections
    assert only.text == "first second third line"
    assert (only.start_seconds, only.end_seconds) == (1.0, 12.0)


def test_text_outside_ascii_is_kept_as_written():
    [only] = transcript_windows(_events(_event(0, "café — 水 🌊 naïve")))
    assert only.text == "café — 水 🌊 naïve"


def test_a_video_longer_than_an_hour_is_titled_with_hours():
    [section] = transcript_windows(_events(_event(3_675_000, "late", duration_ms=5_000)))
    assert section.title == "1:01:15–1:01:20"
    assert videomod.clock(59.9) == "0:59"
    assert videomod.clock(3600) == "1:00:00"


def test_a_window_is_75_seconds():
    events = _events(*[_event(second * 1000, f"w{second}") for second in range(0, 300, 10)])
    assert len(transcript_windows(events)) == 4


def test_a_time_too_large_for_a_float_is_not_a_time():
    # json reads an integer of any length; a float cannot hold one past ~10^308.
    huge = "9" * 400
    with pytest.raises(ValueError, match="caption file"):
        transcript_windows(
            json.loads(f'{{"events": [{{"tStartMs": {huge}, "segs": [{{"utf8": "x"}}]}}]}}')
        )
    event = f'{{"tStartMs": 1000, "dDurationMs": {huge}, "segs": [{{"utf8": "x"}}]}}'
    [only] = transcript_windows(json.loads(f'{{"events": [{event}]}}'))
    assert (only.start_seconds, only.end_seconds) == (1.0, 1.0)  # an unusable duration counts as 0


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"events": "none"},
        {"events": [{"tStartMs": "soon", "segs": [{"utf8": "x"}]}]},
        {"events": [{"tStartMs": 0, "segs": "x"}]},
        [],
    ],
)
def test_a_caption_file_of_another_shape_is_refused(bad):
    with pytest.raises(ValueError, match="caption file"):
        transcript_windows(bad)


def test_captions_with_no_text_at_all_are_refused():
    with pytest.raises(VideoRefusedError, match="no text"):
        transcript_windows(_events({"tStartMs": 0, "id": 1}, _event(1_000, " ")))


# --- what the video says about itself -------------------------------------------


def test_provenance_names_the_channel_the_date_and_the_captions_read():
    info = _info("xAcTmDO6NTI")
    provenance = video_provenance(info, choose_track(info))
    assert provenance == {
        "title": "Lecture 1: Introduction to CS and Programming Using Python",
        "authors": [],
        "publisher": "MIT OpenCourseWare",
        "publication_date": "2024-04-11",
        "doi": None,
        "scholarly": False,
        "video": {
            "id": "xAcTmDO6NTI",
            "channel_id": "UCEBb1b_L6zDS3xTUrIALZOw",
            "channel_verified": True,
            "duration_seconds": 3810,
            "embeddable": True,
            "captions": {"kind": "manual", "language": "en"},
        },
    }


def test_a_channel_is_verified_only_when_youtube_says_so():
    # yt-dlp reports None, not False, for a channel without the badge.
    info = _info("HBtdbaSKexU")
    assert info["channel_is_verified"] is None
    assert video_provenance(info, choose_track(info))["video"]["channel_verified"] is False


@pytest.mark.parametrize(
    ("fields", "embeddable"),
    [
        ({"playable_in_embed": True, "age_limit": 0}, True),
        ({"playable_in_embed": False, "age_limit": 0}, False),
        ({"playable_in_embed": True, "age_limit": 18}, False),
        ({"playable_in_embed": None, "age_limit": None}, False),
    ],
)
def test_a_video_is_embeddable_only_when_youtube_allows_it_for_everyone(fields, embeddable):
    info = {**_info("HBtdbaSKexU"), **fields}
    assert video_provenance(info, choose_track(info))["video"]["embeddable"] is embeddable


def test_long_self_descriptions_are_cut():
    info = {**_info("HBtdbaSKexU"), "title": "t" * 5000, "channel": "c" * 5000}
    provenance = video_provenance(info, choose_track(info))
    assert len(provenance["title"]) == len(provenance["publisher"]) == videomod.MAX_METADATA_CHARS


def test_a_missing_or_malformed_upload_date_is_no_date():
    for value in (None, "", "2016", "yesterday", "20161399"):
        info = {**_info("HBtdbaSKexU"), "upload_date": value}
        assert video_provenance(info, choose_track(info))["publication_date"] is None, value


# --- refusals -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (
            # Captured in the spike, from two live streams.
            "ERROR: [youtube] M3HKLzjvKPc: Sign in to confirm you’re not a bot. Use "
            "--cookies-from-browser or --cookies for the authentication. See  "
            "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for "
            "how to manually pass cookies.",
            "YouTube asked to confirm the reader is not a bot (it asks this of live streams, and "
            "when it limits requests from this network) — try again later",
        ),
        (
            "ERROR: [youtube] abcdefghijk: Private video. Sign in if you've been granted access "
            "to this video",
            "Private video. Sign in if you've been granted access to this video",
        ),
        (
            "ERROR: [youtube] abcdefghijk: Video unavailable. This video has been removed by "
            "the uploader",
            "Video unavailable. This video has been removed by the uploader",
        ),
        (
            "ERROR: [youtube] abcdefghijk: Sign in to confirm your age. This video may be "
            "inappropriate for some users. Use --cookies-from-browser or --cookies for the "
            "authentication.",
            "Sign in to confirm your age. This video may be inappropriate for some users.",
        ),
        (
            "ERROR: unable to download webpage: HTTP Error 429: Too Many Requests",
            "unable to download webpage: HTTP Error 429: Too Many Requests",
        ),
    ],
)
def test_youtubes_own_reason_is_quoted_without_the_command_line_advice(message, reason):
    assert reason_from_ytdlp(message, "abcdefghijk") == reason


def test_a_reason_is_quoted_but_bounded():
    reason = reason_from_ytdlp("ERROR: [youtube] abcdefghijk: " + "x" * 5000, "abcdefghijk")
    assert len(reason) < 300 and reason.endswith("...")


# --- the reader's options -------------------------------------------------------


def test_the_reader_options_confine_yt_dlp_to_one_video_on_youtube(monkeypatch):
    monkeypatch.setattr(videomod, "_deno_path", lambda: "/opt/deno")
    options = video_options(logger=object())
    assert options["allowed_extractors"] == ["youtube"]
    assert options["noplaylist"] is True
    assert options["skip_download"] is True
    assert options["extractor_args"] == {"youtube": {"skip": ["hls", "dash"]}}
    assert options["proxy"] == ""  # a direct connection, never an environment proxy
    assert options["cachedir"] is False  # nothing written to disk
    assert options["js_runtimes"] == {"deno": {"path": "/opt/deno"}}
    assert options["remote_components"] == set()  # no solver code fetched from the network
    assert options["socket_timeout"] == videomod.SOCKET_TIMEOUT_SECONDS
    # Captions are what is wanted: with these set, yt-dlp reports captions it
    # withheld for want of a token as a warning instead of a debug line
    # (nothing is written: skip_download, and extract_info never processes).
    assert options["writesubtitles"] is True and options["writeautomaticsub"] is True
    assert options["retries"] == options["extractor_retries"] == 1
    assert "cookiefile" not in options and "cookiesfrombrowser" not in options
    assert options.get("nocheckcertificate") is not True


@pytest.mark.parametrize("missing", ["binary", "package"])
def test_a_missing_javascript_runtime_is_the_readers_failure_naming_the_video(monkeypatch, missing):
    import sys

    import deno

    def no_binary():
        raise FileNotFoundError("/env/bin/deno")

    if missing == "binary":
        monkeypatch.setattr(deno, "find_deno_bin", no_binary)
    else:
        monkeypatch.setitem(sys.modules, "deno", None)  # import deno now raises ImportError
    with pytest.raises(videomod.VideoReaderError) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=FakeYDL())
    assert str(info.value) == (
        f"The YouTube video {CANONICAL} could not be read: the video reader's JavaScript "
        "runtime (deno) is not installed — reinstall the backend's pinned dependencies "
        f"(pip install -e .) (yt-dlp {videomod._ytdlp_version()})"
    )


def test_a_missing_yt_dlp_is_the_readers_failure_naming_the_video(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", None)
    with pytest.raises(videomod.VideoReaderError) as info:
        videomod.read_video("HBtdbaSKexU")
    assert str(info.value) == (
        f"The YouTube video {CANONICAL} could not be read: yt-dlp is not installed — reinstall "
        "the backend's pinned dependencies (pip install -e .)"
    )


def test_the_server_imports_neither_yt_dlp_nor_deno():
    """Both load only in the reader process: a server whose environment lacks
    them still starts and reads PDFs, and a video read fails naming the video."""
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, authorai.jobs, authorai.api, authorai.video as v; v._ytdlp_version; "
            "print('yt_dlp' in sys.modules, 'deno' in sys.modules)",
        ],
        cwd=BACKEND,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split() == ["False", "False"]


# --- reading one video (yt-dlp replaced by a recording stand-in) ------------------

CANONICAL = "https://www.youtube.com/watch?v=HBtdbaSKexU"


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: answers extract_info with `info` (or
    raises `error`), serves `caption` from urlopen, and records every call.
    `warnings` are logged through the options' logger during extraction, as
    yt-dlp logs them."""

    def __init__(self, info=None, caption=b"", error=None, warnings=(), open_error=None):
        self.info, self.caption, self.error, self.warnings = info, caption, error, warnings
        self.open_error = open_error
        self.params: dict | None = None
        self.extracted: list[tuple] = []
        self.opened: list[str] = []

    def __call__(self, params):
        self.params = params
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True, process=True):
        self.extracted.append((url, download, process))
        for warning in self.warnings:
            self.params["logger"].warning(warning)
        if self.error is not None:
            raise self.error
        return json.loads(json.dumps(self.info))

    def urlopen(self, url):
        self.opened.append(url)
        if self.open_error is not None:
            raise self.open_error
        import io

        return io.BytesIO(self.caption)


@pytest.fixture()
def deno_found(monkeypatch):
    monkeypatch.setattr(videomod, "_deno_path", lambda: "/opt/deno")


def _nasa(**fields) -> dict:
    return {**_info("HBtdbaSKexU"), **fields}


def test_a_captioned_video_is_read_into_sections_and_provenance(deno_found):
    ydl = FakeYDL(_nasa(), MANUAL_BYTES)
    document, provenance = videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    # The reader is handed the canonical link it built, never the one pasted.
    assert ydl.extracted == [(CANONICAL, False, False)]
    assert ydl.opened == ["https://www.youtube.com/api/timedtext?v=x&lang=en&fmt=json3"]
    assert document.title == "ScienceCasts: The Power of Light"
    assert document.tables == [] and document.figures == []
    assert [s.start_seconds // 75 for s in document.sections] == [0, 1, 2, 3]
    assert provenance["publisher"] == "NASA Science"
    assert provenance["video"]["captions"] == {"kind": "manual", "language": "en"}


def test_a_video_without_usable_captions_is_refused_naming_it(deno_found):
    ydl = FakeYDL(_info("C1wFmXGPbUg"))
    with pytest.raises(VideoRefusedError) as info:
        videomod.read_video("C1wFmXGPbUg", ydl_class=ydl)
    assert str(info.value) == (
        "The YouTube video https://www.youtube.com/watch?v=C1wFmXGPbUg cannot be read: it has "
        "no usable captions (neither captions written for it nor YouTube's automatic captions "
        "of the language it is spoken in)"
    )
    assert ydl.opened == []


def test_yt_dlp_itself_reports_withheld_captions_to_the_readers_log(deno_found):
    """Not a stand-in: yt-dlp's own YouTube extractor, given the reader's
    options, routes its "missing subtitles ... PO token" notice to the log as a
    warning — the path the refusal below reads."""
    import yt_dlp

    log = videomod._YtDlpLog()
    with yt_dlp.YoutubeDL(video_options(log)) as ydl:
        extractor = ydl.get_info_extractor("Youtube")
        extractor._report_pot_subtitles_skipped(
            "C1wFmXGPbUg",
            True,
            msg="C1wFmXGPbUg: There are missing subtitles languages because a PO token was not "
            "provided.",
        )
    assert any("PO token" in warning for warning in log.warnings)


@pytest.mark.parametrize(
    "warning",
    [
        "[youtube] C1wFmXGPbUg: There are missing subtitles languages because a PO token was not "
        "provided.",
        "[youtube] C1wFmXGPbUg: Some web client subtitles require a PO Token which was not "
        "provided.",
    ],
)
def test_captions_withheld_in_either_wording_are_not_called_missing(deno_found, warning):
    ydl = FakeYDL(_info("C1wFmXGPbUg"), warnings=[warning])
    with pytest.raises(VideoRefusedError, match="YouTube withheld this video's captions"):
        videomod.read_video("C1wFmXGPbUg", ydl_class=ydl)


class _HTTPError(Exception):
    status = 429


def test_a_failed_caption_download_is_the_readers_failure_named_once(deno_found):
    ydl = FakeYDL(
        _nasa(), open_error=_HTTPError("https://www.youtube.com/api/timedtext?ip=1.2.3.4")
    )
    with pytest.raises(videomod.VideoReaderError) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    message = str(info.value)
    assert message.startswith(f"The YouTube video {CANONICAL} could not be read: its captions ")
    assert "(_HTTPError, HTTP 429)" in message
    assert message.count("could not be read") == 1
    assert "1.2.3.4" not in message  # the caption address carries the reader's IP
    assert videomod._failure_kind(info.value) == "failed"


def test_the_reader_cuts_a_long_transcript_and_records_the_span_it_kept(deno_found):
    ydl = FakeYDL(_nasa(), MANUAL_BYTES)
    whole, provenance = videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    assert "truncated" not in provenance
    kept_two = len(whole.sections[0].text) + len(whole.sections[1].text)
    ydl = FakeYDL(_nasa(), MANUAL_BYTES)
    document, provenance = videomod.read_video("HBtdbaSKexU", ydl_class=ydl, max_chars=kept_two)
    assert document.sections == whole.sections[:2]
    total = sum(len(section.text) for section in whole.sections)
    assert provenance["truncated"] == {
        "kept_chars": kept_two,
        "dropped_chars": total - kept_two,
        "kept_until_seconds": whole.sections[1].end_seconds,
    }


def test_a_window_cut_by_the_cap_claims_no_span(deno_found):
    """A first window longer than the cap is cut mid-window, and how far into
    the video the cut falls is not known: the record keeps the two numbers only."""
    ydl = FakeYDL(_nasa(), MANUAL_BYTES)
    document, provenance = videomod.read_video("HBtdbaSKexU", ydl_class=ydl, max_chars=100)
    assert len(document.sections) == 1 and len(document.sections[0].text) <= 100
    assert set(provenance["truncated"]) == {"kept_chars", "dropped_chars"}


def _reader_that_is_terminated(sender, _payload_path, pid_file, cpu_seconds):
    import subprocess
    import sys
    import threading

    from authorai import web

    web.end_with_parent(cpu_seconds, end_group_on_sigterm=True)
    videomod.prepare_reader_process()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(pid_file).write_text(str(sleeper.pid))
    threading.Event().wait(60)


_EXITING_SERVER = """
import contextlib, logging, sys, threading
from pathlib import Path
import authorai.web as web
from tests.test_video import _reader_that_is_terminated
# Quiet: the reader thread is a daemon still running as the interpreter ends,
# and a daemon thread writing to stderr then makes Python abort the exit.
logging.disable(logging.CRITICAL)
# The reader thread may or may not reach its own cleanup before the interpreter
# ends (a race it won in a trial run): take that cleanup out, so the reader's
# own handler is what is tested.
web._kill_group = lambda pid: None
pid_file = Path(sys.argv[1])
def read():
    with contextlib.suppress(Exception):
        web.in_bounded_child(
            _reader_that_is_terminated, (str(pid_file),), payload_path=None,
            url="https://www.youtube.com/watch?v=HBtdbaSKexU", timeout=60.0,
        )
threading.Thread(target=read, daemon=True).start()
import time
deadline = time.monotonic() + 30
while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
    time.sleep(0.01)
print(pid_file.read_text() if pid_file.exists() else "", flush=True)
sys.exit(0)  # a graceful exit: multiprocessing's exit hook terminates the reader
"""


def test_a_reader_terminated_by_the_servers_exit_takes_deno_with_it(tmp_path):
    """On a graceful stop the server's in_bounded_child never gets to stop the
    reader: multiprocessing's exit hook sends the daemon reader SIGTERM, and
    nothing else runs. The reader's own handler must end its group, or the
    deno it started runs on."""
    import signal
    import subprocess
    import sys

    from tests.test_web_reader import _gone_within, _running

    pid_file = tmp_path / "sleeper.pid"
    output = tmp_path / "server.txt"
    # Files, not pipes: an orphan holding a pipe open would make the run wait
    # for it, and read as the server exiting only when the orphan did.
    with output.open("w") as out:
        server = subprocess.run(
            [sys.executable, "-c", _EXITING_SERVER, str(pid_file)],
            cwd=BACKEND,
            env=_subprocess_env(),
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
    assert server.returncode == 0, output.read_text()
    sleeper = int(pid_file.read_text())
    try:
        assert _gone_within(sleeper, 5), "deno (here a sleeper) outlived its reader"
    finally:
        if _running(sleeper):
            os.kill(sleeper, signal.SIGKILL)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("is_live", "it is live now"),
        ("is_upcoming", "it has not started yet"),
        ("post_live", "it has just ended"),
    ],
)
def test_a_live_or_unfinished_video_is_refused_until_later(deno_found, status, reason):
    ydl = FakeYDL(_nasa(live_status=status))
    with pytest.raises(VideoRefusedError, match=reason) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    assert "try again later" in str(info.value)
    assert ydl.opened == []


def test_a_recording_of_a_past_stream_is_read(deno_found):
    ydl = FakeYDL(_nasa(live_status="was_live"), MANUAL_BYTES)
    document, _ = videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    assert document.sections


def _download_error(message: str, *, expected: bool):
    from yt_dlp.utils import DownloadError, ExtractorError

    cause = ExtractorError(message.split(": ", 2)[-1], expected=expected)
    return DownloadError(message, (ExtractorError, cause, None))


def test_youtubes_refusal_is_a_refusal_naming_the_video(deno_found):
    error = _download_error(
        "ERROR: [youtube] HBtdbaSKexU: Private video. Sign in if you've been granted access "
        "to this video",
        expected=True,
    )
    with pytest.raises(VideoRefusedError) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=FakeYDL(error=error))
    assert str(info.value) == (
        f"The YouTube video {CANONICAL} cannot be read: Private video. Sign in if you've been "
        "granted access to this video"
    )


def test_an_unexpected_extraction_failure_names_the_video_and_the_reader_version(deno_found):
    error = _download_error(
        "ERROR: [youtube] HBtdbaSKexU: Unable to extract initial player response", expected=False
    )
    with pytest.raises(RuntimeError) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=FakeYDL(error=error))
    assert not isinstance(info.value, VideoRefusedError)
    message = str(info.value)
    assert CANONICAL in message
    assert "Unable to extract initial player response" in message
    assert f"yt-dlp {videomod._ytdlp_version()}" in message


def test_a_runtime_yt_dlp_could_not_use_fails_the_read(deno_found):
    # yt-dlp extracts anyway and only warns; formats and captions may be missing.
    ydl = FakeYDL(
        _nasa(),
        MANUAL_BYTES,
        warnings=["[youtube] No supported JavaScript runtime could be found. Only deno is enabled"],
    )
    with pytest.raises(RuntimeError, match="JavaScript runtime"):
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)


def test_a_caption_address_that_is_not_youtubes_is_refused_without_quoting_it(deno_found):
    info = _nasa()
    info["subtitles"]["en"][0]["url"] = "https://evil.example/api/timedtext?ip=203.0.113.9"
    ydl = FakeYDL(info)
    with pytest.raises(RuntimeError) as error:
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    assert "caption" in str(error.value)
    assert "203.0.113.9" not in str(error.value) and "evil.example" not in str(error.value)
    assert ydl.opened == []


def test_captions_over_the_byte_cap_are_refused(deno_found, monkeypatch):
    monkeypatch.setattr(videomod, "CAPTION_MAX_BYTES", 1000)
    ydl = FakeYDL(_nasa(), MANUAL_BYTES)
    with pytest.raises(VideoRefusedError, match="larger than 1,000 bytes"):
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)


def test_a_caption_file_that_is_not_json_is_refused(deno_found):
    ydl = FakeYDL(_nasa(), b"<transcript>not json</transcript>")
    with pytest.raises(ValueError, match="caption file"):
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)


@pytest.mark.parametrize("fields", [{"_type": "playlist"}, {"_type": "url"}, {"id": "aaaaaaaaaaa"}])
def test_anything_but_the_one_video_asked_for_is_refused(deno_found, fields):
    ydl = FakeYDL(_nasa(**fields), MANUAL_BYTES)
    with pytest.raises(RuntimeError, match="not the video"):
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)


def test_the_reader_is_given_only_a_video_id():
    with pytest.raises(ValueError, match="video id"):
        videomod.read_video("../../x", ydl_class=FakeYDL())


# --- the reader process ----------------------------------------------------------


def test_the_readers_environment_keeps_only_what_a_program_needs():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/Users/x",
        "TMPDIR": "/tmp/x",
        "LANG": "en_GB.UTF-8",
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "OPENAI_API_KEY": "sk-secret",
        "AUTHORAI_API_KEY": "app-secret",
        "HTTPS_PROXY": "http://proxy.internal:3128",
        "DENO_DIR": "/somewhere",
        "PYTHONPATH": "/x",
    }
    videomod.scrub_environment(environ)
    assert environ == {
        "PATH": "/usr/bin",
        "HOME": "/Users/x",
        "TMPDIR": "/tmp/x",
        "LANG": "en_GB.UTF-8",
        "YTDLP_NO_PLUGINS": "1",
    }


def _answers(*addresses):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            (address, 443),
        )
        for address in addresses
    ]


def test_the_reader_may_resolve_only_youtube_and_only_to_public_addresses():
    asked = []

    def resolver(host, port, *args, **kwargs):
        asked.append(host)
        return {
            "www.youtube.com": _answers("142.250.184.206", "2a00:1450:4003:80e::200e"),
            "youtube.com": _answers("10.0.0.7"),
        }[host]

    guarded = videomod.guarded_getaddrinfo(resolver)
    assert guarded("www.youtube.com", 443) == resolver("www.youtube.com", 443)
    assert guarded(b"WWW.YOUTUBE.COM.", "443")  # bytes, capitals, a trailing dot
    for host, port in [
        ("example.org", 443),
        ("i.ytimg.com", 443),
        ("www.youtube.com", 80),
        ("169.254.169.254", 443),
    ]:
        asked.clear()
        with pytest.raises(socket.gaierror, match="only YouTube"):
            guarded(host, port)
        assert asked == []  # refused before any lookup
    with pytest.raises(socket.gaierror, match="private or reserved"):
        guarded("youtube.com", 443)


def _reader_reports_its_world(sender, _payload_path, cpu_seconds):
    """A reader process prepared as the video reader prepares one, reporting
    what it can see: its environment and whether a non-YouTube host resolves."""
    from authorai import web

    web.end_with_parent(cpu_seconds)
    videomod.prepare_reader_process()

    def world():
        try:
            socket.getaddrinfo("example.org", 443)
            resolved = True
        except socket.gaierror:
            resolved = False
        return dict(os.environ), resolved

    web.report_outcome(sender, world, lambda exc: "other")


def test_a_reader_process_holds_no_keys_and_resolves_nothing_but_youtube(monkeypatch):
    from authorai.web import in_bounded_child

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-planted")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-planted")
    kind, (environ, resolved) = in_bounded_child(
        _reader_reports_its_world, (), payload_path=None, url=CANONICAL, timeout=30
    )
    assert kind == "result"
    assert "ANTHROPIC_API_KEY" not in environ and "OPENAI_API_KEY" not in environ
    assert not any("planted" in value for value in environ.values())
    assert environ["YTDLP_NO_PLUGINS"] == "1"
    assert resolved is False


@pytest.mark.parametrize(
    ("outcome", "error", "message"),
    [
        (
            ("refused", ("VideoRefusedError", "The YouTube video x cannot be read: no", "tb")),
            VideoRefusedError,
            "The YouTube video x cannot be read: no",
        ),
        (("value", ("ValueError", "bad caption file", "tb")), ValueError, "bad caption file"),
        (
            ("other", ("KeyError", "'boom'", "tb")),
            RuntimeError,
            f"{CANONICAL} could not be read: KeyError: 'boom'",
        ),
        # The reader's own failure already names the video and yt-dlp's version:
        # passed on as it is, not wrapped in a second "could not be read".
        (
            ("failed", ("VideoReaderError", "The YouTube video x could not be read: y", "tb")),
            videomod.VideoReaderError,
            "The YouTube video x could not be read: y",
        ),
    ],
)
def test_the_readers_failures_come_back_as_their_own_kind(monkeypatch, outcome, error, message):
    monkeypatch.setattr(videomod, "in_bounded_child", lambda *a, **k: outcome)
    with pytest.raises(error) as info:
        videomod.read_video_bounded("HBtdbaSKexU", timeout=5, max_chars=5000)
    assert type(info.value) is error
    assert str(info.value) == message


def test_the_readers_own_failures_are_told_apart_from_unexpected_ones(deno_found):
    ydl = FakeYDL(_nasa(_type="playlist"))
    with pytest.raises(videomod.VideoReaderError) as info:
        videomod.read_video("HBtdbaSKexU", ydl_class=ydl)
    assert isinstance(info.value, RuntimeError)
    assert videomod._failure_kind(info.value) == "failed"
    assert videomod._failure_kind(KeyError("x")) == "other"


def test_the_bounded_reader_returns_what_the_reader_read(monkeypatch):
    sections = transcript_windows(MANUAL)
    read = ParsedDocument(title="Title", sections=sections, tables=[], figures=[])
    result = ("result", (read, {"publisher": "NASA Science"}))
    seen = {}

    def fake(target, args, **kwargs):
        seen.update(target=target, args=args, **kwargs)
        return result

    monkeypatch.setattr(videomod, "in_bounded_child", fake)
    document, provenance = videomod.read_video_bounded("HBtdbaSKexU", timeout=90, max_chars=5000)
    assert document.title == "Title" and document.sections == sections
    assert provenance == {"publisher": "NASA Science"}
    assert seen["args"] == ("HBtdbaSKexU", 5000)
    assert seen["payload_path"] is None
    assert seen["url"] == CANONICAL and seen["timeout"] == 90
