# src/analysis/accuracy/

Prediction accuracy comparison between Polymarket and FiveThirtyEight against FEC ground truth.

## Scripts

### `accuracy.py`

Compares **Polymarket** (prediction market) and **FiveThirtyEight** (poll aggregator) against **FEC ground truth** for the 2024 U.S. presidential election.

```
python -m src.analysis.accuracy.accuracy
```

Compatibility alias (legacy):

```
python -m src.analysis.accuracy
```

Raw output is saved in [`result.txt`](result.txt).

#### Input Data

All files are read from `data/processed/`:

| File | Source | Description |
|---|---|---|
| `polymarket_daily.csv` | Polymarket | Daily win-probability per state (Mar 8 – Nov 4, all 50 states) |
| `polls_538.csv` | FiveThirtyEight | Daily poll-average vote-share per state (Mar 1 – Sept 12, 15 states) |
| `fec_results.csv` | FEC | Official 2024 results: vote counts, margins, winner, electoral votes per state |

#### Key Constraints

- **538 data ends September 12, 2024.** All head-to-head comparisons use that date as the cutoff.
- **13 overlap states** have data in both sources on Sept 12: AZ, CA, FL, GA, MI, MN, NC, NH, NV, OH, PA, TX, WI. This is the fair comparison set.
- **Polymarket duplicates** are handled upstream in `src/clean/clean_polymarket.py` (keeps the latest `timestamp_utc` per state per date). The processed CSV is already clean.
#### Analysis Sections

**1. Winner Prediction Accuracy**

For each source, the predicted winner per state is determined by a simple threshold:
- Polymarket: `trump_prob > 0.5` → Trump, else Harris
- 538: `trump_pct > dem_pct` → Trump, else Harris

Results are compared against `fec_results.winner` and reported as correct/total with individual misses listed. Snapshots:
- Sept 12 head-to-head (13 states, both sources)
- Swing-state-only subset (7 states in overlap)
- Polymarket standalone at Oct 6 (30 days out, 50 states)
- Polymarket standalone at Nov 4 (election eve, 50 states)

**Key findings from Section 1:**

- **MI and WI were uniquely stubborn.** They appear as misses in every single snapshot: both sources on Sept 12, Polymarket at 30 days out, and Polymarket on election eve. No other states were missed that consistently. These were the two narrowest Trump swing-state wins — both sources sat right at the decision boundary and fell on the wrong side.
- **The market largely settled by 30 days out.** Polymarket went from 47/50 on Oct 6 to 48/50 on Nov 4, picking up just one additional state in the final month. The two it couldn't crack (MI, WI) stayed wrong through election eve. This suggests the market had priced in most available information by early October, and the remaining uncertainty was concentrated in the tightest races.

**2. Electoral Vote Predictions**

Maps each source's predicted state winners to FEC `electoral_votes` and sums Trump vs Harris EV totals. Reported for:
- Sept 12 head-to-head (13 states, 248 EV in sample)
- Polymarket Nov 4 standalone (all 50 states, 535 EV)

Compared against the actual result: Trump 312 – Harris 226.

**3. Time-Series Accuracy (March – Sept 12)**

A daily loop over the full overlap period. For each date:
- Polymarket uses the day's row per state
- 538 uses forward-fill (latest available row per state on or before that date)
- Winner predictions are made for the common states and checked against FEC

Reports:
- Mean and median daily accuracy for each source
- Period breakdown: Mar–May, Jun–Jul, Aug–Sep 12

**Key finding — the crossover pattern:**

Polymarket's lower overall time-series accuracy (74.9% vs 538's 90.2%) is heavily driven by early-period noise. The period breakdown tells a more nuanced story:

- **Mar–May:** 538 was nearly perfect (99.4%) while Polymarket sat at 63.6%. Prediction markets need trading volume and information to settle; 538's polling model was stable and largely reflecting incumbency/fundamentals.
- **Jun–Jul:** Polymarket climbed to 87.7% as liquidity and attention grew.
- **Aug–Sep 12:** The relationship flipped — Polymarket (79.2%) outperformed 538 (62.2%). This was the period after Biden dropped out and Harris entered the race, which scrambled polling averages. Polymarket adapted faster because traders incorporate new information instantly, while polling models need new polls to trickle in.

This is a classic finding in prediction-market research: markets are noisier early on when there is less signal to trade on, but outperform polling models at incorporating late-breaking information. The crossover happens precisely when a major shock (candidate swap) hit the race — the exact scenario where markets' speed advantage matters most.

#### Functions

| Function | Visibility | Purpose |
|---|---|---|
| `load_data()` | Public | Load 3 CSVs, parse dates, drop 538 national rows |
| `_latest_per_state(p538, as_of)` | Private | Forward-fill: latest 538 row per state ≤ date |
| `_predict_winner_pm(df)` | Private | Add `predicted_winner` from Polymarket probabilities |
| `_predict_winner_538(df)` | Private | Add `predicted_winner` from 538 vote-share |
| `_winner_accuracy(snapshot, fec, label, states)` | Private | Count and print correct winner calls |
| `_ev_prediction(snapshot, fec, label)` | Private | Sum and print electoral votes by predicted winner |
| `_section_winner(pm, p538, fec)` | Private | Orchestrate Section 1 output |
| `_section_ev(pm, p538, fec)` | Private | Orchestrate Section 2 output |
| `_section_timeseries(pm, p538, fec)` | Private | Orchestrate Section 3 output |
| `main()` | Public | Entry point — load data, run all sections |

#### Constants

| Constant | Value | Meaning |
|---|---|---|
| `OVERLAP_CUTOFF` | `2024-09-12` | Last date with 538 data |
| `ELECTION_DATE` | `2024-11-05` | Election day |
| `THIRTY_DAYS_OUT` | `2024-10-06` | 30 days before election |
| `OVERLAP_STATES` | 13 states | States present in both sources on Sept 12 |
| `SWING_OVERLAP` | 7 states | Swing states within the overlap set |

#### Dependencies

- `pandas`, `numpy` (listed in `requirements.txt`)
- `src.clean.utils` — imports `PROCESSED_DIR` and `SWING_STATES`
