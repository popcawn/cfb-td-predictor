# 🏈 College Football Anytime-TD Simulator

A standalone, single-file web app that predicts **anytime-touchdown scorers** for any FBS game. Pick a game and it
prices **every rostered player** — anytime, **1st TD / Last TD / 2+ TD** and **fair American odds** — plus an **EV
calculator** with a TAKE / PASS verdict, **Your card** (a short, sized bet list), a same-game **parlay picker**, a
**cross-game parlay slip** and a **bet log with CLV**. The college version of the
[NFL Anytime-TD Simulator](https://github.com/popcawn/nfl-td-predictor).

`cfb-td-predictor.html` is fully self-contained (data, model and logos embedded) and runs **offline**. Live extras —
the current line for the whole slate and the game-day forecast — load when online and fall back silently when not.

## On any PC or phone
- **Just use it:** open **https://popcawn.github.io/cfb-td-predictor/** — nothing to install. The data is rebuilt on GitHub
  every Sunday and Thursday (`.github/workflows/refresh.yml`); lines refresh live every visit.
- **Your bet log / slip / bankroll** are saved in the browser you use — move them with ⬇ Export / ⬆ Import.
- **Work on the code:** install [Git](https://git-scm.com) and [Node 18+](https://nodejs.org), then
  `git clone https://github.com/popcawn/cfb-td-predictor.git`. Run `git pull` before each session (the refresh bot
  commits data twice a week). `CLAUDE.md` carries the project notes for Claude Code on any machine.

## Using it
- **This week's games** — the ~70-game FBS slate, filterable by **conference**, **Top 25**, **not started yet** or
  **vs FCS**, plus a team search. One click sets both teams and the line; it opens on the next game to kick off.
  Lines refresh live from ESPN on every visit. A game that has kicked off shows **LIVE** (books pull the line at
  kickoff — type the closing line to price it); the app never prices a game on made-up numbers.
- **Starting QB** — each team has a 🎙️ picker (college QB changes are constant: the tale of the tape flags teams where
  different QBs have led games this season). If the starter is set OUT, the next QB up is promoted.
- **Statuses** — there is **no college injury feed** (ESPN publishes none). The model scales down players who didn't
  touch the ball last game (×0.4), in two-plus games (×0.3) or all season (×0.15) — backtested — but a mid-week injury
  won't show until he misses a game. Set **OUT / Q / DBT** by hand after checking the availability report.
- **Player tags** — usage tier (feature / regular / rotation / deep backup, by touches per game — college has no snap
  counts), ⏸ idle, 🪑 no touches this year, ◦ **no history** (freshman / walk-on / FCS transfer: priced from what such
  players typically produce — treat any "edge" as a guess), ↪ transfer with his previous school, ↩ returner, FR.
- **Low-confidence rosters** get a ⚠ banner with the reasons (QB changes, thin history, FCS team, one game played).
- **Your card, paste box, markets, parlays, slip, bet log** — same as the NFL app: paste FanDuel stacked boards
  (name, then Anytime / 1st / Last); one price per name fills the selected market (use that for the 2+ board).
  Books only list notable players; tick **show the whole roster** to see everyone the model prices.
- **Bet log / slip / bankroll** live in your browser — move them between machines with ⬇ Export / ⬆ Import.

## Files
| File | What it is |
|---|---|
| **`cfb-td-predictor.html`** | The app. Open in any browser. |
| `build-cfb-td-snapshot.mjs` | Node build: data pipeline, backtest, model-switch harness, defense-TD fit. |
| `cfb-model.js` | Every probability formula — shared verbatim by the backtest and the app. |
| `cfb-td-predictor.template.html` | UI source (the build injects the snapshot + model into it). |
| `cfb-td-snapshot.json` | The data snapshot (also embedded in the HTML). |
| `build-market-lines.mjs` / `market_lines_2025.json` | ESPN BET anytime-TD boards (which players the book listed) for the market check. |
| `venues.json` | Geocoded stadiums (for the forecast). |
| `inject.mjs` | Template-only dev loop: re-inject the existing snapshot without rebuilding the data. |
| `_serve.mjs` | Tiny local web server for testing (`node _serve.mjs` → http://localhost:8765). |
| `.github/workflows/refresh.yml` | Scheduled rebuild on GitHub (Sun + Thu). |

## Refreshing the data
```bash
node build-cfb-td-snapshot.mjs                    # seasons 2023-2026, ~1 minute after the first run
BT_EXPERIMENTS=1 node build-cfb-td-snapshot.mjs   # also run the model-switch validation harness
```
Node 18+ and `curl`, no npm packages. The first run downloads ~2 GB of play-by-play into `%TEMP%\cfb_cache`
(completed seasons are cached for good; the current season re-downloads each run).

## Data sources (all free, no API key)
- **[sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data)** GitHub releases — ESPN college
  play-by-play (every FBS game, closing total + spread on each), play participants (TD scorers), game rosters (starter
  flags), season rosters. Player ids are **ESPN athlete ids**, so a transfer's history follows him to his new school.
- **ESPN** — this week's slate and lines, full rosters, AP / CFP rankings, logos.
- **Open-Meteo** — stadium geocoding and forecasts.

## The model
**Team level.** Expected offensive TDs = the team's **Vegas implied total** × **κ** (offensive TDs per point of
closing total, measured: ~0.119), leaning slightly up for big favorites (they convert implied points to TDs better —
measured, +5% per 10 points). Team TD counts are negative binomial.

**Player level.** Each TD goes to a player by opportunity: goal-line carries, carry volume and rushing-TD rate on the
ground; red-zone targets, target volume and receiving-TD rate through the air — recency-weighted across seasons
(this season 1, last 0.3, two ago 0.09) and shrunk toward a position prior for thin samples. College-specific pieces,
each kept only because it improved the backtest on both halves of the season:
- **QB rushing** ×1.4 — college QBs score far more on the ground than their usage profile says.
- **One starting QB**; backups at 0.3 weight (college backups play in blowouts).
- **Availability from usage recency** (no injury feed): idle last game ×0.4, 2+ games ×0.3, none all season ×0.15.
- **Players with no history** get the *measured* production of history-less rostered players (most never touch the
  ball) instead of a replacement-level guess.
- **Deep backups** (<2 touches/game) calibrated ×0.7 (cross-fitted).

**Defense** props pay on **defensive TDs only** (pick-6, fumble return): league rate × e^(0.02 × points favored) ×
(opponent giveaways ÷ league)^0.5, fit on 2024 and tested on 2025. **Kick/punt return TDs** go to the returner's prop.

**Tested and not used:** spreading TDs to the bench as the spread grows ("garbage time" — it only helps if you already
know which backups play; across the whole roster it hurt, and big favorites' starters actually score *more* than the
line alone implies), weather (no reliable effect — shown as context), air yards, catches-instead-of-targets, and a 2+ TD
correction.

## Honesty & calibration
Backtest: trained on 2024, then **every 2025 FBS game (932)** predicted using only data from before it, anchored to
the real closing line, running the app's exact math.
- Players who touched the ball (actives known): **Brier 0.1505 vs 0.1677** base rate → **10.2% skill**, log loss 0.477,
  reliability on the diagonal.
- Whole rosters (what the app prices): 19.9% skill (flattered by many easy zeros).
- On the players ESPN BET actually listed (291 games, weeks 1–9): Brier 0.1756 vs 0.1916 (8.4% skill). The book listed
  only **72% of actual scorers** — college boards skip a lot of players who score.
- Defense TDs: better than league average on both halves of 2025.
- Known limits: 2+ TD runs a little hot in the middle of the range; players with no history are still over-rated
  (1.6% predicted vs 0.6%); backup QBs under-rated (1.5% vs 2.4%).

ESPN archives which players the book listed, **not the prices**, so this shows calibration — not an edge over the
book. The model tracks the market; edge, if any, comes from speed on role/injury news, line shopping and boosts. The
bet log's **CLV** is how you find out. *Not betting advice.*
