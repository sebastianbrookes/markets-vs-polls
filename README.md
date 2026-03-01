# Markets vs. Polls: Forecasting the 2024 U.S. Presidential Election

**Do prediction markets or traditional polls better forecast election outcomes?**

This project compares [Polymarket](https://polymarket.com/) (a prediction market) and [FiveThirtyEight](https://projects.fivethirtyeight.com/polls/president-general/2024/national/) (a polling aggregator) on two dimensions:

1. **Accuracy** — Which source more accurately predicted actual state-level results, as certified by the FEC?
2. **Responsiveness** — How did each source react to major campaign events (Biden's dropout, the assassination attempt, VP picks), and which told a more timely, accurate story?

---

## Data Sources

| Source | Description | Granularity | Link |
|--------|-------------|-------------|------|
| **Polymarket** | Market prices reflecting each candidate's win probability | State-level; daily, hourly, minutely | [Kaggle](https://www.kaggle.com/datasets/pbizil/polymarket-2024-us-election-state-data) |
| **FiveThirtyEight** | Aggregated polling averages by candidate vote share | National + state-level | [GitHub](https://github.com/fivethirtyeight/data/blob/master/polls/2024-averages/presidential_general_averages_2024-09-12_uncorrected.csv) |
| **FEC Official Results** | Certified vote counts by state (ground truth) | State-level | [fec.gov](https://www.fec.gov/documents/5645/2024presgeresults.xlsx) |
| **Campaign Events** | Hand-curated timeline of major campaign moments | Event-level | [Ballotpedia](https://ballotpedia.org/Important_dates_in_the_2024_presidential_race) |

## Methods

The analysis uses three approaches:

- **Snapshot accuracy comparisons** — At fixed points before Election Day, compare each source's state-level predictions against certified FEC results and compute implied Electoral College totals.
- **Time-series analysis** — Track daily swing-state averages from both sources to visualize convergence (or divergence) with actual outcomes over time.
- **Indexed event windows** — Isolate windows around key campaign events to measure the size, direction, and speed of each source's reaction.

## Repository Structure

```
├── data/
│   ├── raw/                        # Untouched source files
│   │   ├── polymarket/             # State-level Kaggle CSVs
│   │   ├── fivethirtyeight/        # Polling averages CSV
│   │   ├── fec/                    # Official results (.xlsx)
│   │   └── events/                 # Campaign events timeline
│   └── processed/                  # Cleaned outputs from src/clean/
│
├── src/
│   ├── clean/                      # Raw → processed transforms
│   ├── analysis/
│   │   ├── accuracy/               # Prediction accuracy comparisons
│   │   └── events/                 # Event-response analysis
│   └── visualize/                  # Plotting functions
│
├── figures/
│   ├── accuracy/                   # Accuracy-related figures
│   └── events/                     # Event-response figures
│
├── notebooks/                      # Exploratory analysis
├── deliverables/                   # Submitted reports
└── requirements.txt
```

## Setup & Reproduction

```bash
git clone <repo-url>
cd markets-vs-polls
pip install -r requirements.txt
```

Run the full pipeline in order:

```bash
python -m src.clean.run_all              # 1. Clean raw data → data/processed/
python -m src.analysis.accuracy          # 2. Run accuracy analysis
python src/visualize/accuracy_plots.py   # 3. Generate accuracy figures
python src/visualize/event_plots.py      # 4. Generate event-response figures
```

## Ethical Considerations

Both data sources carry inherent biases worth noting. Polymarket's user base [skews male](https://www.similarweb.com/website/polymarket.com/#demographics) and presumably more financially secure, meaning market prices may not reflect the beliefs of the broader electorate. FiveThirtyEight's polling data underrepresents harder-to-reach groups like [younger voters](https://www.cbsnews.com/news/youth-vote-hard-to-turn-out-hard-to-poll/) and those with [lower institutional trust](https://publicwise.org/publication/the-problem-of-polling/). The campaign events timeline was hand-curated, introducing subjective judgment about which events "matter."