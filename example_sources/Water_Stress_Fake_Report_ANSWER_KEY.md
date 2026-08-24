# Answer key — Water_Stress_Fake_Report.pdf

Upload `Water_Stress_Fake_Report.pdf` as the **REPORT** and the six PDFs in
`example source two/` as the **SOURCES**. Every planted claim below is
labeled with the verdict the pipeline *should* reach and the stance the
extractor *should* assign. Verdicts are relative to the six sources only.

Written **before** the run (2026-08-14), from a hand audit of the sources —
not from pipeline output.

## Expected headline behavior

- Disavowed-and-contradicted claims count **for** the report; straight-asserted
  fabrications count **against** it. With ~16 supported + ~7 disavowed
  fabrications correct vs ~5 asserted fabrications incorrect, stance-aware
  accuracy should land roughly **0.78–0.86** if the judge performs at its
  measured level (under-commitment C→UNVERIFIABLE downgrades of 1–3 claims
  are within observed variance and lower it slightly; they never raise it).
- The five out-of-scope fabrications belong in **coverage**, not accuracy.
- The UI should show **disavowed badges** on the three prose-disavowed claims
  and the five "Circulating myth (fabricated)" table rows.

## Planted claims

### SUPPORTED + asserted (real facts, stated straight)

| # | Claim | Source (where) |
|---|-------|----------------|
| S1 | 2023 and 2024 were the two warmest years on record | Drought Hotspots p.7 |
| S2 | World 2025 GHI score 18.3, barely improved from 19.0 in 2016 | 2025 GHI p.5 |
| S3 | Current sea-level rise 4.7 mm/yr (last-10-year average) | WMO/WCRP brief p.3–4 |
| S4 | Hunger alarming in 7 countries, serious in 35 | 2025 GHI p.5 |
| S5 | Somalia highest 2025 GHI score, 42.6; held two decades | 2025 GHI p.9/13 (Table 1.1) |
| S6 | Conflict remains the primary driver of global hunger | 2025 GHI p.7/37 |
| S7 | Military spending US$2.7T in 2024, >100× humanitarian allocation | 2025 GHI p.8 |
| S8 | 673 million people (8.2%) undernourished in 2024 | 2025 GHI p.7 (inside the disavowal sentence — its own stance is asserted: the report offers it as the TRUE figure) |
| S9 | 68M people in Southern Africa (17%) needed food aid by late Aug 2024 | Drought Hotspots p.12 |
| S10 | Zambia depends on hydropower for 87% of its electricity | Drought Hotspots p.14 |
| S11 | Panama Canal tolls −US$100M/month; traffic −36% | Drought Hotspots p.29 |
| S12 | Irrigation uses over 70% of the world's developed water | IWMI RR19 Summary |
| S13 | ~50% of the 2025 demand increase can be met by irrigation effectiveness | IWMI RR19 p.17 |
| S14 | IWMI projected 3,625 km³ world withdrawals by 2025 (high-effectiveness) | IWMI RR19 p.13 — attributed relay, no falsity marker → asserted |
| S15 | ABC 92 L/capita/day; realistic indoor 175 L/capita/day | Crouch et al. (AQUA 2021) abstract |
| S16 | 0.5 m sea-level rise by 2100 unavoidable under Paris-consistent pathway | WMO/WCRP brief p.3 |
| S17 | US$1 in nature-based water strategies returns US$27 | Drought Hotspots p.38 |
| S18 | Low hunger may not be reached until 2137 at current pace | 2025 GHI p.5 |
| — | Table REAL column rows (18.3, 7 alarming, 42.6, 4.7 mm/yr, 92 LPCD) duplicate S2/S4/S5/S3/S15 — prose+table duplicate coverage will cost some precision, same as both eval sets | |

### CONTRADICTED + **asserted** (straight fabrications — should be punished)

| # | Claim | Incompatible source statement |
|---|-------|-------------------------------|
| C1 | In 2025 climate volatility displaced conflict as the leading driver of hunger | GHI: "Conflict remains the primary driver of global hunger" |
| C2 | The Panama Canal carries 40% of annual international maritime trade | Drought: "critical for 5 per cent of annual international maritime trade" |
| C3 | Global sea level fell by 2 mm in 2024 as ice sheets stabilised | Brief: sea level "increased by 5.9 mm in 2024"; SLR accelerating |
| C4 | (§2 modelling sentence is one compound with C1 — extractor may emit once) | |
| C5 | (Reserved: the extractor may split §3's Zambezi sentence — see D2) | |

*Note: only 3 clean straight-contradiction targets were planted (C1–C3); the
"~5 incorrect" headline range above also counts judge treatment of the two
weaker fabrications (aquaculture/Singapore) if it finds contradicting
evidence — expected UNVERIFIABLE, see U-list.*

### CONTRADICTED + **disavowed** (fabrications the report itself debunks — should count FOR the report)

| # | Claim (falsity marker) | Incompatible source statement |
|---|------------------------|-------------------------------|
| D1 | Hunger effectively eradicated by 2020 ("That claim is simply false") | GHI: 673M undernourished 2024; score 18.3 moderate |
| D2 | Zambezi reservoirs at normal levels through 2024 ("this is fabricated") | Drought: Kariba at 7% of normal generation |
| D3 | Sea level rising 60 mm/yr ("that figure is invented") | Brief: 4.7 mm/yr current rate |
| T1 | World GHI 2.5, hunger nearly eliminated (fabricated column) | GHI: 18.3 |
| T2 | Only 2 countries remain alarming (fabricated column) | GHI: 7 |
| T3 | Somalia 9.0, reached low hunger 2024 (fabricated column) | GHI: 42.6, highest |
| T4 | Sea level stopped rising in 2020 (fabricated column) | Brief: 4.7 mm/yr, accelerating |
| T5 | 8 L/capita/day suffices for healthy urban living (fabricated column) | Paper: ABC minimum 92; WHO floor 20–50 — borderline: a strict judge may say UNVERIFIABLE if it reads "suffices" as opinion |

### UNVERIFIABLE + asserted (out of the sources' scope — coverage bucket)

| # | Claim | Verified absent from all six sources |
|---|-------|--------------------------------------|
| U1 | Aquaculture supplies 60% of protein in drought-affected regions | fisheries/aquaculture: 0 coverage (GHI + IWMI greps) |
| U2 | Murray-Darling worst drought of the century in 2024 | Australia absent from Drought Hotspots |
| U3 | Aral Sea regained 15% of its 1960 volume | Aral: 0 hits anywhere |
| U4 | Desalination renders irrigation efficiency unnecessary by 2030 ("some commentators claim" — neutral relay, NO falsity marker → stays **asserted** per policy) | desalination not covered |
| U5 | 2026 Global Water Accord: 40 nations halving agricultural withdrawals | invented treaty, no source mentions it |
| U6 | Singapore vertical farms produced 12% of its vegetables in 2024 | vertical farming: 0 hits |

## What to check in the UI

1. Dashboard: disavowed badges on D1–D3 and T1–T5; accuracy ring ≈ 0.8;
   contradicted count ≈ 6–8 with accuracy still high — the metric copy
   explains why.
2. Claims workspace: C1–C3 (asserted+CONTRADICTED) are the report's real
   errors; check their quoted evidence names the right source passage.
3. Chat: ask "which claims are the report's actual mistakes?" — it should
   name C1–C3 and NOT the disavowed ones (the stance legend in the system
   prompt exists for exactly this).
4. Sources panel: the AQUA paper has a DOI (10.2166/aqua.2021.056) and
   should tier VERIFIED_DOI via Crossref; the 1998 IWMI report and 2025 GHI
   have ISBNs, not DOIs — expect VERIFIED_TITLE/METADATA_ONLY; the WSG
   corporate report should tier low (METADATA_ONLY/NONE) — a nice
   credibility-spread check.
