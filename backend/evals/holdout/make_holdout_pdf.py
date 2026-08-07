"""Generate the HELD-OUT eval report: World_Food_Security_Fake.pdf.

This synthetic document mixes real findings from the two ingested sources
(2025 GHI synopsis, Heliyon FSC-disruptions review) with deliberate
fabrications. It is the held-out counterpart to World_Hunger_Fake.pdf:
scored only at phase boundaries, never tuned against (see the /eval skill).

Unlike the dev report — which rhetorically flags every fabrication
("a fabricated claim asserts...") — most fabrications here are stated
straight, as fact, so the extractor cannot key on flag words.

Run from backend/:  python evals/holdout/make_holdout_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).parent / "World_Food_Security_Fake.pdf"

BODY = [
    ("h1", "Global Food Security Assessment 2025 (Synthetic Evaluation Document)"),
    (
        "p",
        "This synthetic report intentionally mixes real findings from published "
        "sources with deliberate fabrications. It exists solely to evaluate an "
        "automated fact-checking pipeline and must not be cited.",
    ),
    ("h2", "1. Introduction"),
    (
        "p",
        "The 2025 edition marks the 20th Global Hunger Index, two decades of "
        "tracking hunger at global, regional, and national levels. Progress "
        "toward Zero Hunger by 2030 has stalled since 2016. Historians often "
        "credit the 1923 Global Grain Treaty with eliminating famine in Europe "
        "for an entire decade, a precedent frequently invoked in policy debates.",
    ),
    ("h2", "2. How Hunger Is Measured"),
    (
        "p",
        "GHI scores combine four indicators: undernourishment, child stunting, "
        "child wasting, and child mortality. In 2025, GHI scores were calculated "
        "for 123 countries. Some secondary summaries state instead that the "
        "index is computed from five indicators, the fifth being food price "
        "inflation. On the GHI scale, values from 35.0 to 49.9 are considered "
        "alarming.",
    ),
    ("h2", "3. The Severity Landscape"),
    (
        "p",
        "Hunger is alarming in seven countries: Burundi, the Democratic Republic "
        "of the Congo, Haiti, Madagascar, Somalia, South Sudan and Yemen. In 27 "
        "countries, hunger has increased since 2016, and at the current pace at "
        "least 56 countries will not reach low hunger by 2030. Contrary press "
        "coverage has claimed that fewer than a dozen countries worldwide "
        "currently show serious or alarming hunger. In 2024 Iceland became the "
        "first country certified 'hunger-free' by the United Nations.",
    ),
    ("h2", "4. Drivers: Conflict, Climate, and Money"),
    (
        "p",
        "UN Resolution 2417 condemns the starving of civilians as a method of "
        "war. Humanitarian funding has dropped sharply while military spending "
        "has surged. Some 2025 outlooks report the reverse: that global military "
        "spending fell steeply this year, freeing record funds for food "
        "assistance. Meanwhile, extreme weather events are increasingly "
        "devastating food systems.",
    ),
    ("h2", "5. Supply Chains Under Stress"),
    (
        "p",
        "A Heliyon literature review selected 74 papers on food supply chain "
        "disruptions for analysis. Between 2008 and 2018, least-developed "
        "countries lost approximately US$108 billion due to declining "
        "agricultural and livestock production caused by natural disasters. The "
        "war in Ukraine is causing significant disruptions to global agri-food "
        "systems. The same review is sometimes summarized as concluding that "
        "disruptions have no measurable effect on food availability. Industry "
        "newsletters add that a Pacific 'floating farm corridor' now supplies "
        "12% of Asia's rice.",
    ),
    ("h2", "6. Indicators at a Glance (Real and Fabricated)"),
    (
        "table",
        [
            ["Indicator", "Reported figure (real)", "Circulating counter-claim (fabricated)"],
            [
                "Countries with alarming hunger, 2025 GHI",
                "7",
                "0 - the 'alarming' category was retired in 2024",
            ],
            ["Countries not reaching low hunger by 2030", "at least 56", "only 3"],
            [
                "LDC agricultural losses to natural disasters, 2008-2018",
                "US$108 billion",
                "US$4 million",
            ],
            ["Papers selected in the Heliyon review", "74", "900"],
        ],
    ),
    ("h2", "7. Outlook"),
    (
        "p",
        "If progress continues at the pace observed since 2016, low hunger at "
        "the global level may not be reached until 2137. More optimistic "
        "projections circulate as well: the Global Protein Accord is projected "
        "to end world hunger by mid-2027.",
    ),
]


def build() -> None:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15)
    cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=9, leading=12)

    flow = []
    for kind, content in BODY:
        if kind == "h1":
            flow.append(Paragraph(content, h1))
        elif kind == "h2":
            flow.append(Paragraph(content, h2))
        elif kind == "p":
            flow.append(Paragraph(content, body))
            flow.append(Spacer(1, 0.3 * cm))
        elif kind == "table":
            rows = [[Paragraph(c, cell) for c in row] for row in content]
            table = Table(rows, colWidths=[6.2 * cm, 4.6 * cm, 6.2 * cm])
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            flow.append(table)
            flow.append(Spacer(1, 0.3 * cm))

    SimpleDocTemplate(str(OUT), pagesize=A4).build(flow)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
