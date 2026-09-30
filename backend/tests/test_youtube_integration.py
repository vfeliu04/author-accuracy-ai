"""The live video reader: yt-dlp and deno against YouTube itself, in the
bounded reader process with its scrubbed environment and guarded DNS.

Local only (`-m integration`): it needs the network, and YouTube changes
under yt-dlp every few months — this is the smoke test of the bump routine
(docs/development.md). The two videos are licensed CC BY by their uploaders
(tests/fixtures/youtube/README.md) and have been stable for years.
"""

import pytest

from authorai.video import VideoRefusedError, read_video_bounded

pytestmark = pytest.mark.integration


def test_a_video_with_written_captions_is_read_whole():
    document, provenance = read_video_bounded("HBtdbaSKexU", timeout=120)
    assert document.title == "ScienceCasts: The Power of Light"
    assert provenance["publisher"] == "NASA Science"
    assert provenance["publication_date"] == "2016-12-13"
    assert provenance["video"]["captions"] == {"kind": "manual", "language": "en"}
    assert provenance["video"]["embeddable"] is True
    text = " ".join(section.text for section in document.sections)
    assert "Science@NASA Crewmembers on the International Space Station" in text
    assert [section.start_seconds // 75 for section in document.sections] == [0, 1, 2, 3]


def test_a_dubbed_video_is_read_from_its_own_languages_speech():
    # 18 speech-recognition tracks, one per dubbed audio track, Arabic listed
    # first: the English one is read, because the original audio is English.
    document, provenance = read_video_bounded("BVVzkThVMg4", timeout=120)
    assert provenance["video"]["captions"] == {"kind": "automatic", "language": "en"}
    assert document.sections[0].text.startswith("As we all know, this morning we're gathered")


def test_a_video_without_captions_is_refused_naming_it():
    with pytest.raises(VideoRefusedError, match="C1wFmXGPbUg cannot be read: it has no usable"):
        read_video_bounded("C1wFmXGPbUg", timeout=120)
