# College Football (FBS) Anytime-TD Simulator — working notes

College version of `C:\Users\User\nfl-td-predictor` (github.com/popcawn/nfl-td-predictor). Same shape: a single
self-contained HTML app + a Node build script that bakes the data in. Reuse that project's architecture, UI and
rules; read its CLAUDE.md first. The user gates each phase: 0 feasibility, 1 build + snapshot, 2 model + backtest
3 UI port and 4 GitHub are done (2026-09-25). Repo github.com/popcawn/cfb-td-predictor (public), live at
https://popcawn.github.io/cfb-td-predictor/ (Pages serves `main`).
See README.md for what the app does and how it was validated.

## Where things happen
- `build-cfb-td-snapshot.mjs` — data pipeline + backtest + switch harness + defense-TD fit. Writes `cfb-td-snapshot.json` and,
  once the template exists, injects it into `cfb-td-predictor.template.html` → `cfb-td-predictor.html`.
- `cfb-model.js` — **all probability math** (NB anytime, expected team TDs, player scoring channels, run/pass split,
  defensive-TD rate). The build `new Function`-loads it for the live scores AND the backtest, and injects the same
  source into the app at `/*__MODEL__*/`. Change a probability here, behind a `MODEL` switch — never in one place only.
- `cfb-td-predictor.template.html` — the UI (ported from the NFL template). It never computes a probability itself:
  `sideModel()` builds each player's inputs and calls `CFBModel.distribute()`; the displayed anytime and 2+ are that
  closed form (+ `CFBModel.roleCal`), and the seeded Monte Carlo (NB team counts, TDs handed out by the same shares)
  supplies 1st/last TD and parlay joints — checked to match the closed form within MC noise. Teams are keyed by ESPN
  team id (not abbreviation). `window.cfbDebug()` returns the current run state for console checks.
- `venues.json` — geocoded stadium coordinates (Open-Meteo), keyed by ESPN venue id. Committed so geocoding rarely reruns.
- Never hand-edit `cfb-td-snapshot.json` / `cfb-td-predictor.html`; the build regenerates both.
- **Data refresh runs on GitHub Actions** (`.github/workflows/refresh.yml`, Sun + Thu 15:00 UTC, + Run workflow button)
  and commits the rebuilt html/json/venues — so **`git pull` before starting work**, or pushes conflict on those files.
  The runner restores a fixed-key cache of completed seasons (`cfb-history-2026-v1`); **bump the key when a new season
  starts** so the season that just finished gets cached. `gh` isn't installed: read run status from
  https://api.github.com/repos/popcawn/cfb-td-predictor/actions/runs (public).

## Build / dev loop
- `node build-cfb-td-snapshot.mjs` (Node 18+, curl). ~1 min. First run downloads ~2 GB into `%TEMP%\cfb_cache`
  (override `CFB_CACHE_DIR`); completed seasons are cached forever, the current season re-downloads every run.
- Dev loop: `NO_REFRESH=1 SKIP_LOGOS=1 node build-cfb-td-snapshot.mjs` (~25 s, no current-season re-download).
- Default seasons 2023 2024 2025 2026: newest = current, newest-1 = backtest test, newest-2 = train (2023 = extra
  low-weight history + lets the build tell who was genuinely new in 2024). `BT_EXPERIMENTS=1` runs the switch harness.
- Template-only change: `node inject.mjs` re-injects the existing snapshot + model and parse-checks every script.
- Test in a browser over http, not file://: `node _serve.mjs` (port 8765); the preview tool's config lives
  at `C:UsersUser.claudelaunch.json` (name `cfb-td`). Screenshots often time out — use DOM/JS checks.

## Data sources (keyless)
- sportsdataverse GitHub releases (`https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>`):
  `espn_cfb_pbp` (play-by-play, ESPN athlete ids, closing total/spread), `espn_cfb_play_participants` (TD scorer id),
  `espn_cfb_game_rosters` (per-game starter flags: none in 2024, ~43% of 2025 games, ~85% of 2026),
  `espn_cfb_rosters` (season positions / class), `cfb_schedules` (neutral site, kickoff, venue), `espn_cfb_player_box`
  (only for the TD data check).
- ESPN site/core API: FBS membership by conference (core `seasons/{y}/types/2/groups/80/children`), scoreboard
  `?groups=80&limit=300&week=&seasontype=` (slate + DraftKings line), `/rankings` (AP, CFP), `/teams`, rosters.
- Open-Meteo geocoding (venue city/state → lat/lon) and, in the app, forecasts. CFBD is NOT used (user's call).

## Gotchas (all handled in the build — keep them handled)
- **ESPN college rosters truncate at 100** unless `/roster?limit=300`.
- **ESPN `homeTeamSpread` sign is inverted**; use `homeFavorite` + `|gameSpread|`. PBP lines are closing lines.
- **Per-play score columns are unreliable on ~2% of scoring plays** (stale or scrambled). TDs are identified by
  `scoringPlay` + TD type (or TD text on 'Unknown' plays), minus "nullified"/"no play", de-duplicated on
  game|kind|scorer|period|clock|text (text alone merged two real identical-text TDs). Scoring side comes from play
  semantics (offense for rush/rec; defense for pick-6/fumble return; `return_team` for returns). The build's TD check vs
  ESPN box scores must stay ≈99%+ (2024 99.2%, 2025 99.3%, 2026 99.6%).
- `pass_td` is also true on pick-sixes; `pass_attempt`/`rush` flags: sacks are not rushes (correct).
- Negative athlete ids (`-5866`) are ESPN placeholders for unknown players — ignored.
- **No college injury data or depth charts exist free.** Availability = usage recency (`lastTouchAgo`, `gCur`,
  `touchShare`, starter flags). Game rosters list the whole roster (~112/game), so "dressed" means nothing.
- 2024 incompletions rarely name the receiver (targets ≈ catches); air yards 0% / 43% / 98% in 2024 / 25 / 26.
- Each game's QB = ESPN-flagged starter (two flagged → more 1st-half dropbacks), else the 1st-half dropback leader.
  Garbage-time dropbacks must not make a backup "the QB". `rosterConf: 'low'` + `confNotes` mark shaky rosters.
- FCS teams appear in ESPN PBP only vs FBS; FCS-only games are excluded. NDSU/Sac State etc. change division —
  FBS membership is per season, never hard-coded.
- In Git Bash, `node -e` with backticks gets mangled; long multi-pattern replace scripts are fragile — use the Edit tool.

## Validation record (Phase 2, 2026-09-25) — train 2024, test all 932 FBS games of 2025
- Backtest = `runBacktest()` in the build: rolling, leak-free, closing lines, the app's exact `CFBModel.distribute()`.
  Candidate sets: **touched** (players who touched the ball — actives known, comparable to the NFL headline) and
  **full** (+ the team's whole skill roster — what the app prices, since college has no injury feed).
- Switch rule: `BT_EXPERIMENTS=1`; a flip must beat the current model by ≥0.00003 Brier on BOTH halves (wk 1–7, 8+).
  Depth/availability switches are judged on **full**; they always "win" on touched because touched already knows who played.
- Headline: touched Brier 0.1505 vs 0.1677 base (10.2% skill), log loss 0.477, reliability on the diagonal (low bin
  5→9%); full 0.0526 vs 0.0657 (19.9%, inflated by easy zeros — quote touched); market universe (ESPN BET boards,
  291 games wk 1–9) 0.1756 vs 0.1916 (8.4%), book listed only 72% of actual scorers (~15 players/game).
- Kept: measured new-player prior (`newPrior:'emp'`), usage-recency availability (aNone 0.15 / a1 0.4 / a2 0.3),
  kSlope 0.05 (big favorites convert implied points to TDs better), QB rushing ×1.4 (+ qbCarry 1.3), backup QB 0.3,
  NB shape 5, role calibration for the 'fringe' tier only (×0.7, cross-fitted).
- Rejected: garbage-time flattening (helps only if you know which backups play; hurts the full roster — and big
  favorites' starters actually score MORE than the base model said), weather (within noise → off, shown as context),
  targets-vs-catches, air yards, other wPrior/shrink/eps values, a 2+ TD factor (cross-fit disagrees by set).
- QB check: starters 29.6% pred vs 29.9% actual; backups 1.5% vs 2.4% (still under). 2+ TD runs hot in mid bins on
  touched (24→19%, 34→22%); 1st TD mean 4.0% vs 4.0%. No-history players still over-predicted on full (1.6 vs 0.6%).
- Defense TDs: fit on 2024 → slope 0.02, giveaway exponent 0.5; on 2025 Brier 0.0810 vs league-average 0.0828
  (better in both halves) vs NFL parameters 0.0874 (worse than average). Live base refitted with the multipliers.
- `market_lines_2025.json` comes from `node build-market-lines.mjs 2025` (boards only exist for weeks 1–9).

## App gotchas (Phase 3)
- Books pull lines at kickoff: an in-progress game shows LIVE and the app refuses to price without a total (never
  fall back to made-up defaults — the whole model hangs off the Vegas total). Opens on the next game WITH a line.
- A hand-picked starting QB is available by definition (gamesAgo forced to 0) — otherwise a QB who hasn't played this
  season gets the ×0.15 idle factor and reads ~1%.
- Return-only players (DBs who return kicks) have no offensive touches → not 'no history'; tagged ↩ returner only.
- Wide tables sit in `.tscroll` so the page stays phone-width (a 980px table was widening the mobile layout viewport).
- Storage keys `cfbtd_bets` / `cfbtd_slip` / `cfbtd_bankroll`; export app id `cfb-td-predictor`; bets/slip legs store
  `gameLbl` because next week's snapshot may not contain last week's FCS opponent.

## Rules carried over from the NFL build (non-negotiable)
- One shared `MODEL` config drives both the live app and the backtest; no live-only tweaks.
- Rolling, leak-free backtest (train 2024, test every 2025 FBS game with only prior data, real closing lines).
- A switch stays only if it improves Brier on BOTH halves (weeks 1–7 and 8+; bowls = week 20). Small per-group gaps are noise.
- Separately validate the defense-TD model on held-out data. Never claim edges the backtest doesn't show.
- Ask before creating the GitHub repo, pushing, anything paid, or signing up for anything.
