# YouTube caption fixtures

Real captures made with yt-dlp 2026.8.19 on 2026-09-30 (the PR D spike). Caption URLs are not kept: they carry the requester's IP address and a signature.

- `HBtdbaSKexU.en.json3`: the manual English captions of "ScienceCasts: The Power of Light" by NASA Science (https://www.youtube.com/watch?v=HBtdbaSKexU), whole. Licensed CC BY (Creative Commons Attribution) by its uploader.
- `BVVzkThVMg4.en-orig.json3`: YouTube's automatic (speech-recognition) English captions of "STS-26 SSIP Briefing" by the NASA STI Program (https://www.youtube.com/watch?v=BVVzkThVMg4), events of the first three minutes. Licensed CC BY by its uploader.
- `tracks.json`: for eight public videos, the caption-track keys, the language and `language_preference` of the audio formats, and the metadata fields yt-dlp returns with `process=False`. Two of them (`BVVzkThVMg4`, `_ketUOAj30k`) are auto-dubbed, so they carry one `-orig` track per dubbed audio track.
