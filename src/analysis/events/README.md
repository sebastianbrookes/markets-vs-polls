# Event Response Analysis

## What This Does

When a major campaign event happens — a debate, an assassination attempt, a candidate dropping out — how quickly and accurately do different forecasting sources react?

This analysis compares **Polymarket** (a prediction market) against **FiveThirtyEight** (a polling aggregator) by measuring how each source responded to five key moments in the 2024 presidential race.

## The Five Events

| Event | Date | Expected Effect |
|---|---|---|
| Biden-Trump Debate | June 27 | Pro-Trump (Biden performed poorly) |
| Trump Assassination Attempt | July 13 | Pro-Trump (sympathy rally) |
| Biden Drops Out | July 21 | Pro-Harris (stronger replacement candidate) |
| Walz VP Pick | Aug 6 | Neutral (safe, low-impact pick) |
| Harris-Trump Debate | Sept 10 | Pro-Harris (Harris widely seen as winning) |

## How We Measure Reactions

For each event, the script looks at the **swing-state average** Trump lead from both sources and computes three values:

- **Baseline** — the average over the 3 days *before* the event
- **Immediate** — the value on the day of the event (and the next day)
- **Settled** — the average over 2–7 days *after* the event

The **shift** (settled minus baseline) tells us the net impact. A positive shift means the source moved toward Trump; negative means toward Harris.

Each shift is also converted to a **z-score** — how large the move was relative to that source's normal day-to-day noise. This lets us fairly compare Polymarket (probability scale) and 538 (vote-share % scale).

## Key Findings

**Polymarket: 4/5 correct direction** — reacted in the expected direction for every event except the Walz VP pick (overreaction).

**FiveThirtyEight: 3/5 correct** — matched expectations for the debate and assassination attempt, but moved the wrong way after the Harris-Trump debate and also overreacted to the Walz pick.

Both sources had full data coverage for all five events, though 538 had a 5-day data gap around the final debate.

## Outputs

- `result.txt` — full printed scorecard with numeric details
- `figures/events/timeline.png` — time-series plot with event bands
- `figures/events/scoreboard.png` — bar chart comparing reaction intensity
- `figures/events/dropout-response.png` — before/after comparison for the dropout event

## How to Run

From the project root:

```
python -m src.analysis.events.event_response   # prints scorecard
python -m src.visualize.event_plots             # generates plots
```
