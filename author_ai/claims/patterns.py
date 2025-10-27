"""Centralised regular expression patterns used across the claim pipeline."""

from __future__ import annotations

import regex as re

QUALIFIER_WORDS = (
    "about",
    "approximately",
    "around",
    "roughly",
    "nearly",
    "almost",
    "circa",
    "some",
)

QUALIFIER_PATTERN = re.compile(
    rf"\b(?:{'|'.join(QUALIFIER_WORDS)})\b", flags=re.IGNORECASE
)

NUMBER_WORD_PATTERN = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "sixty|seventy|eighty|ninety|hundred"
)

NUMBER_PATTERN = re.compile(
    r"""
    (?P<number>
        [+-]?
        (?:
            (?:\d{1,3}(?:[,\s]\d{3})+|\d+)
            (?:\.\d+)?|
            \d*\.\d+
        )
    )
    """,
    flags=re.VERBOSE,
)

LOCALE_NUMBER_PATTERN = re.compile(
    r"""
    (?P<number>
        [+-]?
        (?:
            \d{1,3}(?:\.\d{3})*(?:,\d+)? |
            \d{1,3}(?:,\d{3})*(?:\.\d+)? |
            \d+,\d+
        )
    )
    """,
    flags=re.VERBOSE,
)

STATISTIC_PATTERN = re.compile(
    rf"""
    (?P<qualifier>\b(?:about|around|approximately|roughly|nearly|almost|some|circa)\b\s+)?
    (?P<number>
        [+-]?
        (?:
            (?:\d{{1,3}}(?:[,\s]\d{{3}})+|\d+)(?:[.,]\d+)? |
            \d*\.\d+ |
            (?:{NUMBER_WORD_PATTERN})
        )
    )
    (?P<unit>
        \s*
        (?:
            %|percent|percentage\s+points|pp|ppt|
            per\s+\d+[^\s,.;)]*|
            million|billion|thousand|k|m
        )
    )?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

RANGE_PATTERN = re.compile(
    r"""
    (?P<prefix>\b(?:between|from|around|about|approximately|roughly|circa|~)?\b\s*)?
    (?P<start>[+-]?(?:\d+(?:[.,]\d+)?))
    \s*
    (?:
        (?P<dash>[-–—])|
        \s*(?:to|and)\s+
    )
    \s*
    (?P<end>[+-]?(?:\d+(?:[.,]\d+)?))
    \s*
    (?P<unit>
        %|percent|percentage\s+points|pp|
        (?:per\s+\d+[^\s,.;)]*)|
        [a-zA-Z]+
    )?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

RATIO_PATTERN = re.compile(
    r"""
    \b
    (?P<num>(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|
              twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|
              twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))
    \s+(?:in|out\sof)\s+
    (?P<den>(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|
              twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|
              twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

DELTA_PATTERN = re.compile(
    r"""
    (?P<direction>\b(?:up|down|increase(?:d)?|rise|fall|fell|decrease(?:d)?)\b)
    [\s,]+
    (?:(?:by|of)\s+)?
    (?P<value>[+-]?(?:\d+(?:[.,]\d+)?))
    \s*
    (?P<unit>%|percent|percentage\s+points|pp|ppt)?
    (?:\s+(?:vs|versus|than|from|since|over|compared\s+with)\s+
        (?P<baseline>[^.;,]+?))?
    (?=[.;,)]|\s|$)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

QUARTER_PATTERN = re.compile(
    r"""
    (?:
        (?P<qprefix>\bq(?:uarter)?)(?P<qnum>[1-4])\s*(?P<year>\d{2,4}) |
        (?P<year_leading>\d{4})\s*(?:q|quarter)\s*(?P<qnum_leading>[1-4]) |
        (?P<compact>[1-4])q(?P<compact_year>\d{2,4})
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


__all__ = [
    "QUALIFIER_WORDS",
    "QUALIFIER_PATTERN",
    "NUMBER_WORD_PATTERN",
    "NUMBER_PATTERN",
    "LOCALE_NUMBER_PATTERN",
    "STATISTIC_PATTERN",
    "RANGE_PATTERN",
    "RATIO_PATTERN",
    "DELTA_PATTERN",
    "QUARTER_PATTERN",
    "YEAR_PATTERN",
]
