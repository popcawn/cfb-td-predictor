#!/usr/bin/env node
/*
 * build-cfb-td-snapshot.mjs
 * -------------------------------------------------------------------------
 * Builds the offline data snapshot for cfb-td-predictor.html (college football, FBS).
 *
 * Pipeline:
 *   1. FBS membership per season (ESPN core API, by conference) — who counts as FBS changes every year.
 *   2. sportsdataverse releases (keyless GitHub downloads, cached): ESPN play-by-play, play participants (TD
 *      scorers), game rosters (starter flags), season rosters (positions/class) and schedules (neutral site,
 *      kickoff, venue). Player IDs are ESPN athlete IDs everywhere, so a transfer's history follows his ID.
 *   3. Stream-parse the play-by-play into per-player-game and per-team-game usage records, every TD with its
 *      scorer and kind, and each game's closing total/spread.
 *   4. Derive league conversion constants, kappa (offensive TDs per point of closing total), the defensive-TD
 *      model inputs, recency-weighted team profiles and player scores (cfb-model.js — the same math the app and
 *      the backtest use).
 *   5. ESPN live: this week's FBS slate + lines, AP/CFP ranks, team colors + small logos, full rosters
 *      (`?limit=300` — the default silently stops at 100), venue coordinates (Open-Meteo geocoding, cached in
 *      venues.json).
 *   6. Join live rosters to play-by-play by athlete ID, report the join rate and unmatched producers, flag thin
 *      histories (freshmen, transfers, FCS), and write cfb-td-snapshot.json (+ inject into the template if present).
 *
 *   7. Backtest (rolling, leak-free): train on CUR-2, predict every CUR-1 FBS game using only data from before it,
 *      anchored to its real closing line; plus the defense-TD model fit/test and the model-switch harness.
 *
 * Usage:
 *   node build-cfb-td-snapshot.mjs                        # default seasons 2023 2024 2025 2026
 *   node build-cfb-td-snapshot.mjs 2022 2023 2024 2025    # newest = current, newest-1 = backtest test, newest-2 = train
 *   SKIP_LOGOS=1   skip logo downloads           NO_REFRESH=1   reuse cached current-season files (dev loop)
 *   BT_EXPERIMENTS=1  also run the model-switch validation harness (every switch scored on both season halves)
 *   CFB_CACHE_DIR  cache location (default %TEMP%/cfb_cache; ~2 GB for four seasons)
 *
 * No npm dependencies. Requires Node 18+ and curl on PATH.
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { execFileSync, spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
const argSeasons = process.argv.slice(2).map(Number).filter(n => n >= 2004 && n <= 2100);
const SEASONS = (argSeasons.length ? argSeasons : [2023, 2024, 2025, 2026]).sort((a, b) => a - b);
const CUR = Math.max(...SEASONS);
// Backtest: TRAIN (CUR-2) supplies kappa, league constants, priors and the defense fit; TEST (CUR-1) is predicted game by
// game. Older seasons only add low-weight player history (the live model blends the same number of seasons).
const TEST_SEASON = SEASONS.includes(CUR - 1) ? CUR - 1 : null;
const TRAIN_SEASON = SEASONS.includes(CUR - 2) ? CUR - 2 : null;
const CACHE_DIR = process.env.CFB_CACHE_DIR || path.join(process.env.TEMP || process.env.TMP || '/tmp', 'cfb_cache');
fs.mkdirSync(CACHE_DIR, { recursive: true });
const SDV = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download';
const ESPN = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football';
const CORE = 'https://sports.core.api.espn.com/v2/sports/football/leagues/college-football';
const SIM_META = { nsims: 10000 };
const POST_WK = 20;           // bowls/playoff get week 20 so they sort after the regular season (ESPN restarts at 1)

// The shared model math (also injected into the app). Loaded as a plain script so both sides run identical code.
const MODEL_SRC = fs.readFileSync(path.join(__dirname, 'cfb-model.js'), 'utf8');
const CFBModel = new Function(MODEL_SRC + '\nreturn CFBModel;')();

// The model's tunable choices — read by BOTH the live scores and the backtest (via cfb-model.js), and shipped to the
// app. Every value is picked by the switch harness (BT_EXPERIMENTS=1): a change stays only if it improves Brier on
// BOTH halves of the test season.
const MODEL = {
  noKneel: true,       // drop kneel-downs from carry volume
  weather: false,      // wind / rain / snow lean-run on the run/pass TD split — within noise on college data (off; shown as context)
  wPrior: 0.3,         // last season's weight vs this season's (season weights = wPrior^age)
  qbCarry: 1.3,        // QB scaling of the carry-volume term (college QBs run a lot)                          [helps H1+H2]
  qbRush: 1.4,         // QB scaling of his whole rushing weight                                                [helps H1+H2]
  eps: 0.01,           // small universal score floor
  shrinkGames: 4,      // pseudo-games blended into every per-game rate
  shrinkPrior: true,   // shrink thin samples toward the position prior instead of toward zero
  ps: 0.5,             // strength of that prior
  recVol: 'tgt',       // pass-volume signal: 'tgt' (targets) or 'rec' (catches — 2023-24 rarely name incomplete targets)
  air: false,          // air-yards term (0% of 2024 targets, 43% of 2025, 98% of 2026 have air yards)
  // shrinkage target per position (per-game scoring weights)
  prior: { RB: { rush: 0.055, rec: 0.03 }, WR: { rush: 0.004, rec: 0.045 }, TE: { rush: 0.002, rec: 0.035 }, QB: { rush: 0.06, rec: 0 }, X: { rush: 0.002, rec: 0.004 } },
  newPrior: 'emp',     // [helps H1+H2, full roster] players with NO history: 'fixed' = the prior above; 'emp' = measured production of history-less
                       //   rostered players in the train season (most never touch the ball)
  nbSize: 5,           // player-TD overdispersion (gamma-Poisson shape); the app's sim uses the same value   [helps H1+H2 touched]
  twoCal: 1,           // 2+ TD factor — cross-fit disagrees by candidate set (x0.85 touched, x1.1 full), so none
  newEps: true,        // give no-history players the universal floor too (false hurt the known-actives set)
  kSlope: 0.05,        // kappa lean per 10 points favored (big favorites turn implied points into TDs more)  [helps H1+H2]
  qbBackup: 0.3,       // weight of a QB who isn't the presumed starter (college backups play in blowouts)   [helps H1+H2]
  avail: true,         // [helps H1+H2, full + roster] usage-recency availability (no college injury feed): no touch last game -> a1, 2+ games -> a2,
  availGrace: 2,       //   none yet this season once the team has played availGrace games -> aNone
  aNone: 0.15, a1: 0.4, a2: 0.3,   // [aNone/a1 help H1+H2 on the full roster]
  priorScale: 1,       // scales the shrinkage target (prior) — the NFL-sized prior may inflate deep backups
  gt: 0,               // garbage time: flatten a team's TD shares toward the depth chart as |spread| grows past gtFrom
  gtFrom: 7, gtSide: 'both',
  roleCalTiers: ['fringe'],   // usage tiers whose probabilities get the cross-fitted actual/predicted correction [fringe helps H1+H2]
};
const SEASON_WEIGHT = {};
for (const yr of SEASONS) SEASON_WEIGHT[yr] = Math.pow(MODEL.wPrior, CUR - yr);

// Venues whose roof makes weather irrelevant (fixed or retractable). ESPN's indoor flag is also honored.
const DOME_RE = /dome|superdome|alamodome|ford field|mercedes-benz stadium|at&t stadium|lucas oil|nrg stadium|allegiant|state farm stadium|u\.s\. bank stadium|jma wireless|holt arena|tacoma/i;
const STATE_NAME = { AL: 'Alabama', AK: 'Alaska', AZ: 'Arizona', AR: 'Arkansas', CA: 'California', CO: 'Colorado', CT: 'Connecticut', DE: 'Delaware', DC: 'District of Columbia', FL: 'Florida', GA: 'Georgia', HI: 'Hawaii', ID: 'Idaho', IL: 'Illinois', IN: 'Indiana', IA: 'Iowa', KS: 'Kansas', KY: 'Kentucky', LA: 'Louisiana', ME: 'Maine', MD: 'Maryland', MA: 'Massachusetts', MI: 'Michigan', MN: 'Minnesota', MS: 'Mississippi', MO: 'Missouri', MT: 'Montana', NE: 'Nebraska', NV: 'Nevada', NH: 'New Hampshire', NJ: 'New Jersey', NM: 'New Mexico', NY: 'New York', NC: 'North Carolina', ND: 'North Dakota', OH: 'Ohio', OK: 'Oklahoma', OR: 'Oregon', PA: 'Pennsylvania', RI: 'Rhode Island', SC: 'South Carolina', SD: 'South Dakota', TN: 'Tennessee', TX: 'Texas', UT: 'Utah', VT: 'Vermont', VA: 'Virginia', WA: 'Washington', WV: 'West Virginia', WI: 'Wisconsin', WY: 'Wyoming' };

// ----------------------------------------------------------------------------
// Small utilities
// ----------------------------------------------------------------------------
const log = (...a) => console.log(...a);
const num = v => { const n = +v; return Number.isFinite(n) ? n : 0; };
const T = v => v === 'true' || v === 'TRUE' || v === 'True' || v === '1';
const r3 = x => Math.round(x * 1000) / 1000;
const r4 = x => Math.round(x * 10000) / 10000;
const pct = (a, b) => (b ? (a / b * 100).toFixed(1) : '-') + '%';
function curlToFile(url, dest) {
  const tmp = dest + '.part';
  execFileSync('curl', ['-sL', '--fail', '--retry', '3', '-m', '900', '-o', tmp, url], { stdio: 'ignore' });
  fs.renameSync(tmp, dest);
}
function curlJson(url) {
  const r = spawnSync('curl', ['-sL', '--fail', '--compressed', '--retry', '2', '-m', '90', url], { encoding: 'utf8', maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error('curl failed: ' + url);
  return JSON.parse(r.stdout);
}
function curlBase64(url) {
  const r = spawnSync('curl', ['-sL', '--fail', '-m', '60', url], { encoding: 'buffer', maxBuffer: 1 << 26 });
  if (r.status !== 0 || !r.stdout || !r.stdout.length) return null;
  return r.stdout.toString('base64');
}
async function fetchJson(url, tries = 3) {
  for (let a = 0; a < tries; a++) {
    try { const r = await fetch(url); if (r.ok) return await r.json(); if (r.status === 404 || r.status === 400) return null; } catch { }
    await new Promise(res => setTimeout(res, 500 * (a + 1)));
  }
  return null;
}
async function pool(items, n, fn) {   // run fn over items with at most n in flight
  const out = new Array(items.length); let i = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); } }));
  return out;
}
// Quote-aware CSV line splitter (handles "" escapes and quoted commas).
function splitCSV(line) {
  const out = []; let cur = '', inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQ) { if (c === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else inQ = false; } else cur += c; }
    else if (c === '"') inQ = true; else if (c === ',') { out.push(cur); cur = ''; } else cur += c;
  }
  out.push(cur);
  return out;
}
// Stream a CSV (plain or .gz) as {header index, row arrays}.
async function* readCSV(file) {
  let input = fs.createReadStream(file);
  if (file.endsWith('.gz')) input = input.pipe(zlib.createGunzip());
  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  let ix = null;
  for await (const line of rl) {
    if (!ix) { ix = {}; splitCSV(line).forEach((h, i) => { ix[h] = i; }); continue; }
    if (line) yield [ix, splitCSV(line)];
  }
}
// Download a sportsdataverse release asset into the cache. Completed seasons are static (downloaded once);
// the current season changes daily, so it is re-downloaded every build unless NO_REFRESH is set.
function sdvFile(tag, name, season) {
  const dest = path.join(CACHE_DIR, `${tag}__${name}`);
  const have = fs.existsSync(dest) && fs.statSync(dest).size > 1000;
  if (have && (season !== CUR || process.env.NO_REFRESH)) return dest;
  try { log(`  downloading ${tag}/${name}${season === CUR ? ' (current season — refreshed every run)' : ''} ...`); curlToFile(`${SDV}/${tag}/${name}`, dest); }
  catch { if (!have) { log(`  ! ${tag}/${name} not available`); return null; } log(`  ! refresh of ${name} failed; using the cached copy`); }
  return dest;
}
function normName(s) {
  return String(s || '').toLowerCase().replace(/[.'`’]/g, '').replace(/\b(jr|sr|ii|iii|iv|v)\b/g, '').replace(/[^a-z]/g, '').trim();
}
const posBucket = p => { p = (p || '').toUpperCase(); return p === 'RB' || p === 'FB' || p === 'HB' ? 'RB' : p === 'WR' ? 'WR' : p === 'TE' ? 'TE' : p === 'QB' ? 'QB' : null; };

// ----------------------------------------------------------------------------
// FBS membership per season (ESPN core API: FBS = group 80, its children are the conferences)
// ----------------------------------------------------------------------------
const FBS = {};            // season -> Set(teamId)
const CONF_OF = {};        // season -> {teamId: confId}
const CONF_NAME = {};      // confId -> {abbr, name}
async function loadFBS(season) {
  const cache = path.join(CACHE_DIR, `fbs_${season}.json`);
  let data = null;
  if (season !== CUR && fs.existsSync(cache)) data = JSON.parse(fs.readFileSync(cache, 'utf8'));
  else {
    const kids = await fetchJson(`${CORE}/seasons/${season}/types/2/groups/80/children?limit=50`);
    if (!kids) throw new Error('FBS conference list unavailable for ' + season);
    const confIds = kids.items.map(i => i.$ref.match(/groups\/(\d+)/)[1]);
    data = { confs: {}, teams: {} };
    await pool(confIds, 6, async cid => {
      const g = await fetchJson(`${CORE}/seasons/${season}/types/2/groups/${cid}`);
      const t = await fetchJson(`${CORE}/seasons/${season}/types/2/groups/${cid}/teams?limit=60`);
      data.confs[cid] = { abbr: (g && (g.shortName || g.abbreviation)) || cid, name: (g && g.name) || cid };
      for (const it of (t && t.items) || []) data.teams[it.$ref.match(/teams\/(\d+)/)[1]] = cid;
    });
    if (Object.keys(data.teams).length < 100) throw new Error(`FBS team list for ${season} looks wrong (${Object.keys(data.teams).length})`);
    fs.writeFileSync(cache, JSON.stringify(data));
  }
  FBS[season] = new Set(Object.keys(data.teams));
  CONF_OF[season] = data.teams;
  for (const [cid, c] of Object.entries(data.confs)) CONF_NAME[cid] = c;
}

// ----------------------------------------------------------------------------
// Season rosters (positions / class year / team per season) — sportsdataverse espn_cfb_rosters
// ----------------------------------------------------------------------------
const athPos = new Map();     // athleteId -> {season: pos}
const athName = new Map();    // athleteId -> display name
const seasonRoster = {};      // season -> teamId -> Set(skill-position athleteIds) — the backtest's "whole roster" set
async function loadSeasonRosters(season) {
  const f = sdvFile('espn_cfb_rosters', `cfb_rosters_${season}.csv.gz`, season);
  if (!f) return;
  let n = 0;
  const sr = seasonRoster[season] = {};
  for await (const [ix, r] of readCSV(f)) {
    const id = r[ix.athlete_id]; if (!id) continue;
    const pos = r[ix.position_abbreviation];
    let m = athPos.get(id); if (!m) { m = {}; athPos.set(id, m); }
    if (pos) m[season] = pos;
    if (!athName.has(id)) athName.set(id, r[ix.display_name] || r[ix.full_name]);
    if (posBucket(pos)) { const t = r[ix.team_id]; (sr[t] || (sr[t] = new Set())).add(id); }
    n++;
  }
  log(`  ${season} season rosters: ${n} athlete-team rows`);
}
// position bucket as of a season (nearest season at or before it, else any)
function bucketFor(id, season) {
  const m = athPos.get(id); if (!m) return null;
  let best = null, bs = -1;
  for (const [s, p] of Object.entries(m)) { const b = posBucket(p); if (!b) continue; const k = +s <= season ? +s : -1e4 + +s; if (k > bs) { bs = k; best = b; } }
  return best;
}

// ----------------------------------------------------------------------------
// TD scorers (play participants) and starter flags (game rosters)
// ----------------------------------------------------------------------------
const scorerOf = new Map();   // playId -> scorer athleteId
async function loadScorers(season) {
  const f = sdvFile('espn_cfb_play_participants', season === CUR ? `play_participants_${season}.csv` : `play_participants_${season}.csv.gz`, season)
    || sdvFile('espn_cfb_play_participants', `play_participants_${season}.csv`, season);
  if (!f) return;
  for await (const [ix, r] of readCSV(f)) { const s = r[ix.scorer_player_id]; if (s) scorerOf.set(r[ix.play_id], s); }
}
const starters = new Map();   // gameId -> Set(athleteId) — only games where ESPN flagged starters
async function loadStarters(season) {
  const f = sdvFile('espn_cfb_game_rosters', season === CUR ? `game_rosters_${season}.csv` : `game_rosters_${season}.csv.gz`, season);
  if (!f) return;
  let n = 0;
  for await (const [ix, r] of readCSV(f)) {
    if (!T(r[ix.starter])) continue;
    const g = r[ix.game_id]; let s = starters.get(g); if (!s) { s = new Set(); starters.set(g, s); } s.add(r[ix.athlete_id]); n++;
  }
  log(`  ${season} starter flags: ${n} (games with starters listed: ${[...starters.keys()].length})`);
}

// ----------------------------------------------------------------------------
// Schedules (neutral site, kickoff, venue) — sportsdataverse cfb_schedules
// ----------------------------------------------------------------------------
const schedInfo = new Map();  // gameId -> {neutral, kickoff, venueId, venue}
async function loadSchedule(season) {
  const f = sdvFile('cfb_schedules', `cfb_schedules_${season}.csv.gz`, season);
  if (!f) return;
  for await (const [ix, r] of readCSV(f)) schedInfo.set(r[ix.game_id], { neutral: T(r[ix.neutral_site]), kickoff: r[ix.start_date], venueId: r[ix.venue_id], venue: r[ix.venue] });
}

// ----------------------------------------------------------------------------
// Play-by-play
// ----------------------------------------------------------------------------
const games = new Map();      // gid -> {s, wk, st, home, away, total, hMargin, hasLine, hs, as, tds:[], neutral, kickoff}
const teamGames = new Map();  // gid|team -> team-game record
const playerGames = new Map();// pid -> [player-game records]
const pgIndex = new Map();    // pid|gid -> record
const tgPlayers = new Map();  // gid|team -> [player-game records]
const league = {};            // season -> conversion counts
const tdKinds = {};           // season -> {kind: n}  (sanity log)
const kickoffTypes = /^Kickoff|Kickoff Return/;

function teamGame(gid, team, g) {
  const k = gid + '|' + team; let r = teamGames.get(k);
  if (!r) {
    r = { s: g.s, wk: g.wk, gid, team, opp: team === g.home ? g.away : g.home, home: team === g.home, offTD: 0, rushTD: 0, passTD: 0, give: 0,
      plays: 0, carries: 0, tgt: 0, defTD: 0, krTD: 0, prTD: 0, stTD: 0, tdAllow: 0, rushAllow: 0, passAllow: 0, byPos: { RB: 0, WR: 0, TE: 0, QB: 0 } };
    teamGames.set(k, r);
  }
  return r;
}
const DUMMY_PG = { rushAtt: 0, kneel: 0, rushTD: 0, glCarry: 0, tgt: 0, rec: 0, rzTgt: 0, recTD: 0, airY: 0, airTgt: 0, drop: 0, drop1h: 0, kr: 0, pr: 0, stTD: 0 };
const pbpName = new Map();    // athleteId -> name as written in the play-by-play
function playerGame(pid, g, team) {
  if (!pid || pid[0] === '-') { for (const k in DUMMY_PG) DUMMY_PG[k] = 0; return DUMMY_PG; }   // ESPN placeholder ids (unknown athletes)
  const k = pid + '|' + g.gid; let r = pgIndex.get(k);
  if (!r) {
    r = { pid, s: g.s, wk: g.wk, gid: g.gid, team, rushAtt: 0, kneel: 0, rushTD: 0, glCarry: 0, tgt: 0, rec: 0, rzTgt: 0, recTD: 0, airY: 0, airTgt: 0, drop: 0, drop1h: 0, kr: 0, pr: 0, stTD: 0 };
    pgIndex.set(k, r);
    const tk = g.gid + "|" + team; let tl = tgPlayers.get(tk); if (!tl) { tl = []; tgPlayers.set(tk, tl); } tl.push(r);
    let a = playerGames.get(pid); if (!a) { a = []; playerGames.set(pid, a); } a.push(r);
  }
  return r;
}
function tdKind(type, f, ix) {
  if (type === 'Rushing Touchdown') return 'rush';
  if (type === 'Passing Touchdown') return 'rec';
  if (/^Interception Return|^Fumble Return Touchdown|^Fumble Recovery \(Opponent\)/.test(type)) return 'def';
  if (type === 'Kickoff Return Touchdown') return 'kr';
  if (type === 'Punt Return Touchdown') return 'pr';
  if (/Blocked|Missed Field Goal Return|Punt Team Fumble|Kickoff Team Fumble|^Punt|^Kickoff|^Field Goal/.test(type)) return 'st';
  if (type === 'Fumble Recovery (Own) Touchdown') return 'offo';
  if (T(f[ix.int])) return 'def';
  if (T(f[ix.rush])) return 'rush';
  if (T(f[ix.pass_attempt])) return 'rec';
  const text = f[ix.text] || '';                       // 'Unknown' plays: fall back to the play text
  if (/\d+ Yd Run|run for .*for a TD/i.test(text)) return 'rush';
  if (/pass from|pass complete .*for a TD/i.test(text)) return 'rec';
  return 'other';
}

async function parseSeason(season) {
  const f = sdvFile('espn_cfb_pbp', `play_by_play_${season}.csv`, season);
  if (!f) return false;
  log(`  parsing ${season} play-by-play (${(fs.statSync(f).size / 1e6).toFixed(0)} MB) ...`);
  const L = league[season] = { passAtt: 0, passTD: 0, gl: 0, glTD: 0, rzTgt: 0, rzTgtTD: 0, tgt: 0, tgtTD: 0, rec: 0, recTD: 0, carries: 0, kneel: 0, rushTD: 0, airY: 0, airTgt: 0, airTD: 0 };
  const K = tdKinds[season] = {};
  let rows = 0, tdNoSide = 0, tdDupes = 0, tdSideConflict = 0;
  const seenTD = new Set();
  for await (const [ix, r] of readCSV(f)) {
    rows++;
    const st = +r[ix.seasonType]; if (st !== 2 && st !== 3) continue;
    const home = r[ix.homeTeamId], away = r[ix.awayTeamId];
    if (!FBS[season].has(home) && !FBS[season].has(away)) continue;       // FCS-only games are out of scope
    const gid = r[ix.game_id];
    let g = games.get(gid);
    if (!g) {
      const total = num(r[ix.overUnder]), spread = Math.abs(num(r[ix.gameSpread]));
      // ESPN's homeTeamSpread sign is inverted, so use the favorite flag + |spread|. hMargin = points home is favored by.
      const hMargin = spread === 0 ? 0 : (T(r[ix.homeFavorite]) ? spread : -spread);
      const si = schedInfo.get(gid) || {};
      g = { s: season, gid, wk: st === 3 ? POST_WK : +r[ix.week], st, home, away, total, hMargin, hasLine: total > 20, hs: 0, as: 0, tds: [],
        fbsH: FBS[season].has(home), fbsA: FBS[season].has(away), neutral: !!si.neutral, kickoff: si.kickoff || null, venueId: si.venueId || null };
      games.set(gid, g);
      teamGame(gid, home, g); teamGame(gid, away, g);
    }
    const sH0 = num(r[ix['start.homeScore']]), sA0 = num(r[ix['start.awayScore']]);
    const sH1 = num(r[ix['end.homeScore']]), sA1 = num(r[ix['end.awayScore']]);
    if (sH1 > g.hs) g.hs = sH1; if (sA1 > g.as) g.as = sA1;
    const off = r[ix.pos_team_id], def = r[ix.def_pos_team_id];
    const type = r[ix['type.text']];
    const ytg = num(r[ix['start.yardsToEndzone']]);
    const rush = T(r[ix.rush]), kneel = T(r[ix.kneel_down]), passAtt = T(r[ix.pass_attempt]);
    const rid = r[ix.rusher_player_id], cid = r[ix.receiver_player_id], qid = r[ix.passer_player_id];
    const isKick = /Kickoff|Punt|Field Goal|Extra Point/.test(type);

    // ---- team offense usage + giveaways ----
    const to = (off === home || off === away) ? teamGame(gid, off, g) : null;
    if (to && !isKick) {
      if (rush || passAtt) to.plays++;
      if (rush && !kneel) to.carries++;
      if (passAtt && cid) to.tgt++;
      if (T(r[ix.int]) || T(r[ix.fumble_lost])) to.give++;
    }

    // ---- league conversion counts ----
    if (passAtt && !/Sack/.test(type)) { L.passAtt++; if (type === "Passing Touchdown") L.passTD++; }
    if (rush && rid) { if (kneel) L.kneel++; else L.carries++; if (ytg > 0 && ytg <= 5 && !kneel) { L.gl++; if (type === 'Rushing Touchdown') L.glTD++; } if (type === 'Rushing Touchdown') L.rushTD++; }
    if (passAtt && cid) {
      L.tgt++; const td = type === 'Passing Touchdown'; if (td) L.tgtTD++;
      if (/Pass Reception|Passing Touchdown|Pass Completion/.test(type)) { L.rec++; if (td) L.recTD++; }
      if (ytg > 0 && ytg <= 20) { L.rzTgt++; if (td) L.rzTgtTD++; }
      const ay = r[ix.air_yards]; if (ay !== '' && ay !== 'NA') { L.airTgt++; L.airY += Math.max(0, num(ay)); if (td) L.airTD++; }
    }

    // ---- player usage ----
    if (rid && !pbpName.has(rid)) pbpName.set(rid, r[ix.rusher_player_name]);
    if (cid && !pbpName.has(cid)) pbpName.set(cid, r[ix.receiver_player_name]);
    if (qid && !pbpName.has(qid)) pbpName.set(qid, r[ix.passer_player_name]);
    if (rush && rid && to) {
      const p = playerGame(rid, g, off); p.rushAtt++; if (kneel) p.kneel++; else if (ytg > 0 && ytg <= 5) p.glCarry++;
    }
    if (passAtt && cid && to) {
      const p = playerGame(cid, g, off); p.tgt++;
      if (/Pass Reception|Passing Touchdown|Pass Completion/.test(type)) p.rec++;
      if (ytg > 0 && ytg <= 20) p.rzTgt++;
      const ay = r[ix.air_yards]; if (ay !== '' && ay !== 'NA') { p.airTgt++; p.airY += Math.max(0, num(ay)); }
    }
    if (qid && to && (passAtt || /Sack/.test(type))) { const q = playerGame(qid, g, off); q.drop++; if (+r[ix['period.number']] <= 2) q.drop1h++; }   // 1st-half dropbacks = who started
    // kick / punt returns (the returning team is in return_team)
    const krId = r[ix.kickoff_return_player_id], prId = r[ix.punt_return_player_id], retTeam = r[ix.return_team];
    if (krId && (retTeam === home || retTeam === away)) playerGame(krId, g, retTeam).kr++;
    if (prId && (retTeam === home || retTeam === away)) playerGame(prId, g, retTeam).pr++;

    // ---- touchdowns ----
    // ESPN's per-play score columns are stale/scrambled on ~2% of scoring plays, so a TD is identified by the
    // scoringPlay flag + a TD play type (or TD text on an 'Unknown' play), minus penalty-nullified ones, and
    // de-duplicated (ESPN occasionally logs a play twice). Checked against 2025 box scores: 99.3%+ of player TDs.
    const text = r[ix.text] || '';
    const tdText = /for a TD|\d+ Yd (Run|pass|Pass|Interception|Fumble|Kickoff|Punt|Return)/.test(text);
    if (T(r[ix.scoringPlay]) && (/Touchdown/.test(type) || (tdText && !/Field Goal|Safety|Extra Point/.test(type))) && !/nullified|no play/i.test(text)) {
      const kind = tdKind(type, r, ix);
      const scorer = scorerOf.get(r[ix.id]) || (kind === 'rush' ? rid : kind === 'rec' ? cid : kind === 'kr' ? krId : kind === 'pr' ? prId : '');
      const dk = [gid, kind, scorer || '', r[ix['period.number']], r[ix['clock.displayValue']], text.toLowerCase().replace(/\s+/g, ' ').slice(0, 70)].join('|');
      if (seenTD.has(dk)) { tdDupes++; continue; }
      seenTD.add(dk);
      // scoring side from play semantics; the score change only breaks ties (it is unreliable on its own)
      const dH = sH1 - sH0, dA = sA1 - sA0;
      const deltaSide = dH >= 6 && dA < 6 ? home : dA >= 6 && dH < 6 ? away : null;
      const retSide = retTeam === home || retTeam === away ? retTeam : null;
      let side = kind === 'rush' || kind === 'rec' || kind === 'offo' ? off : kind === 'def' ? def : kind === 'kr' || kind === 'pr' || kind === 'st' ? (retSide || def) : deltaSide;
      if (side !== home && side !== away) side = deltaSide;
      if (!side) { tdNoSide++; continue; }
      if (deltaSide && deltaSide !== side) tdSideConflict++;
      K[kind] = (K[kind] || 0) + 1;
      g.tds.push({ team: side, pid: scorer || null, kind, seq: num(r[ix.game_play_number]) || num(r[ix.sequenceNumber]) });
      const ts = teamGame(gid, side, g), ta = teamGame(gid, side === home ? away : home, g);
      if (kind === 'rush' || kind === 'rec' || kind === 'offo') {
        ts.offTD++; if (kind === 'rush') ts.rushTD++; else if (kind === 'rec') ts.passTD++;
        ta.tdAllow++; if (kind === 'rush') ta.rushAllow++; else if (kind === 'rec') ta.passAllow++;
        const bk = scorer && bucketFor(scorer, season); if (bk) ta.byPos[bk]++;
        if (scorer) { const p = playerGame(scorer, g, side); if (kind === 'rush') p.rushTD++; else if (kind === 'rec') p.recTD++; }
      } else if (kind === 'def') ts.defTD++;
      else if (kind === 'kr' || kind === 'pr') { if (kind === 'kr') ts.krTD++; else ts.prTD++; if (scorer) playerGame(scorer, g, side).stTD++; }
      else ts.stTD++;
    }
  }
  for (const x of games.values()) if (x.s === season) x.tds.sort((a, b) => a.seq - b.seq);   // play order (first / last TD)
  const nG = [...games.values()].filter(x => x.s === season).length;
  log(`    ${rows} plays, ${nG} games with an FBS team; TDs by kind ${JSON.stringify(K)}; duplicates dropped ${tdDupes}, side unknown ${tdNoSide}, semantic side != score-change side ${tdSideConflict}`);
  return true;
}

// ----------------------------------------------------------------------------
// Venues (coordinates for weather) — geocoded once via Open-Meteo, cached in venues.json (committed)
// ----------------------------------------------------------------------------
const VENUES_PATH = path.join(__dirname, 'venues.json');
const loadVenues = () => (fs.existsSync(VENUES_PATH) ? JSON.parse(fs.readFileSync(VENUES_PATH, 'utf8')) : {});
const saveVenues = v => fs.writeFileSync(VENUES_PATH, JSON.stringify(v, null, 1));
let geocodedThisRun = 0;
async function ensureVenue(venues, v) {   // v = {id, name, city, state, country, indoor}
  if (!v || !v.id) return null;
  const indoor = !!v.indoor || DOME_RE.test(v.name || '');
  if (venues[v.id] && venues[v.id].lat != null) { venues[v.id].indoor = indoor || venues[v.id].indoor; return venues[v.id]; }
  const us = !v.country || v.country === 'USA';
  const gj = v.city ? await fetchJson(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(v.city)}&count=10${us ? '&countryCode=US' : ''}`) : null;
  const res = (gj && gj.results) || [], hit = res.find(x => !us || x.admin1 === STATE_NAME[v.state]) || res[0];
  if (hit) geocodedThisRun++;
  return (venues[v.id] = { name: v.name, city: v.city, state: v.state, lat: hit ? hit.latitude : null, lon: hit ? hit.longitude : null, indoor });
}
// WMO weather code -> precipitation type (the app maps forecast codes the same way)
const wmoPrecip = c => ([71, 73, 75, 77, 85, 86].includes(c) ? 'snow' : (c >= 51 && c <= 67) || (c >= 80 && c <= 82) || c >= 95 ? 'rain' : 'none');

// Historical game weather for the backtest's weather switch: venue + kickoff from ESPN's scoreboards, observed wind and
// weather code at the kickoff hour from the Open-Meteo archive (one request per date for all that date's venues).
// Cached in the cache dir as wx_<season>.json — completed seasons never change.
async function loadHistWeather(season) {
  const cachePath = path.join(CACHE_DIR, `wx_${season}.json`);
  if (fs.existsSync(cachePath)) return new Map(Object.entries(JSON.parse(fs.readFileSync(cachePath, 'utf8'))));
  log(`  building ${season} game weather (ESPN venues + Open-Meteo archive; cached after this run) ...`);
  const sb = await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&dates=${season}`);
  const info = new Map();
  for (const c of (sb && sb.leagues && sb.leagues[0].calendar) || []) {
    if (c.value !== '2' && c.value !== '3') continue;
    for (const e of c.entries || []) {
      const j = await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&dates=${season}&week=${e.value}&seasontype=${c.value}`);
      for (const ev of (j && j.events) || []) { const v = ev.competitions[0].venue || {}, a = v.address || {};
        info.set(ev.id, { date: ev.date, venue: { id: v.id, name: v.fullName || '', city: a.city || '', state: a.state || '', country: a.country || 'USA', indoor: !!v.indoor } }); }
    }
  }
  const venues = loadVenues();
  for (const x of info.values()) await ensureVenue(venues, x.venue);
  saveVenues(venues);
  const out = {}, byDate = new Map();
  for (const [gid, x] of info) {
    const v = venues[x.venue.id]; if (!v) continue;
    out[gid] = { outdoor: !v.indoor, wind: 0, precip: 'none' };
    if (v.indoor || v.lat == null || !x.date) continue;
    const t = Date.parse(x.date), day = new Date(t).toISOString().slice(0, 10);
    if (!byDate.has(day)) byDate.set(day, []);
    byDate.get(day).push({ gid, lat: v.lat, lon: v.lon, hour: Math.round((t - Date.parse(day + 'T00:00:00Z')) / 3600000) });
  }
  let missing = 0;
  for (const [day, list] of byDate) {
    const end = new Date(Date.parse(day) + 86400000).toISOString().slice(0, 10);
    for (let i = 0; i < list.length; i += 50) {
      const chunk = list.slice(i, i + 50);
      const url = `https://archive-api.open-meteo.com/v1/archive?latitude=${chunk.map(c => c.lat.toFixed(3)).join(',')}&longitude=${chunk.map(c => c.lon.toFixed(3)).join(',')}` +
        `&start_date=${day}&end_date=${end}&hourly=wind_speed_10m,weather_code&wind_speed_unit=mph&timezone=GMT`;
      const j = await fetchJson(url), arr = Array.isArray(j) ? j : j ? [j] : [];
      chunk.forEach((c, k) => { const h = arr[k] && arr[k].hourly; if (!h) { missing++; return; }
        const idx = Math.min(h.time.length - 1, c.hour); out[c.gid].wind = Math.round(h.wind_speed_10m[idx] || 0); out[c.gid].precip = wmoPrecip(h.weather_code[idx]); });
    }
  }
  fs.writeFileSync(cachePath, JSON.stringify(out));
  const vals = Object.values(out);
  log(`    ${vals.length} games: ${vals.filter(x => !x.outdoor).length} indoor, ${vals.filter(x => x.wind > 12).length} with wind > 12 mph, ${vals.filter(x => x.precip === 'rain').length} rain, ${vals.filter(x => x.precip === 'snow').length} snow${missing ? `, ${missing} without archive data` : ''}`);
  return new Map(Object.entries(out));
}


// Data check: rushing + receiving TDs credited from the play-by-play vs ESPN's official player box scores (what
// the books settle on). A drop here means the TD detection/attribution broke upstream — look before trusting a build.
async function checkTDs(season) {
  const f = sdvFile('espn_cfb_player_box', season === CUR ? `player_box_${season}.csv` : `player_box_${season}.csv.gz`, season);
  if (!f) return null;
  const box = new Map();
  for await (const [ix, r] of readCSV(f)) {
    const g = games.get(r[ix.game_id]); if (!g || g.s !== season) continue;
    const n = num(r[ix.rushingTouchdowns]) + num(r[ix.receivingTouchdowns]); if (!n) continue;
    const k = r[ix.athlete_id] + '|' + r[ix.game_id]; box.set(k, (box.get(k) || 0) + n);
  }
  let boxTot = 0, matched = 0, missing = 0, extra = 0;
  for (const [k, n] of box) { const pg = pgIndex.get(k), m = pg ? pg.rushTD + pg.recTD : 0; boxTot += n; matched += Math.min(n, m); if (m < n) missing += n - m; }
  for (const [k, pg] of pgIndex) { if (pg.s !== season) continue; const m = pg.rushTD + pg.recTD; if (!m) continue; const n = box.get(k) || 0; if (m > n) extra += m - n; }
  return { season, boxTDs: boxTot, matched, rate: r4(matched / (boxTot || 1)), missing, extra };
}

// ============================================================================
// Main
// ============================================================================
(async function main() {
  const t0 = Date.now();
  log(`\n=== College football (FBS) anytime-TD snapshot build ===`);
  log(`seasons: ${SEASONS.join(', ')}  |  current ${CUR}  |  backtest train ${TRAIN_SEASON} -> test ${TEST_SEASON}  |  cache ${CACHE_DIR}`);

  for (const s of SEASONS) await loadFBS(s);
  log(`  FBS teams: ${SEASONS.map(s => `${s} ${FBS[s].size}`).join(', ')}`);
  for (const s of [...SEASONS].sort((a, b) => b - a)) await loadSeasonRosters(s);   // positions before bucketing TD scorers
  for (const s of SEASONS) { await loadSchedule(s); await loadScorers(s); }
  for (const s of new Set([TEST_SEASON, CUR])) await loadStarters(s);   // ESPN flags starters from 2025 on
  const gotSeasons = [];
  for (const s of SEASONS) if (await parseSeason(s)) gotSeasons.push(s);
  if (!gotSeasons.includes(CUR)) throw new Error('current season play-by-play missing');
  const fullSeasons = gotSeasons.filter(s => s !== CUR);
  const tdCheck = [];
  for (const s of gotSeasons) { const c = await checkTDs(s); if (c) { tdCheck.push(c); log(`  TD check ${s}: ${c.matched}/${c.boxTDs} box-score rushing+receiving TDs credited to the right player-game (${pct(c.matched, c.boxTDs)}), missing ${c.missing}, extra ${c.extra}`); } }



  // ---- league conversion constants ----
  // TGTTD uses ALL pass attempts as the denominator: 2023-24 incompletions rarely name a receiver, so "TDs per named
  // target" would run hot in those seasons; TDs per attempt is the same quantity without that hole.
  function constFor(seasons) {
    const a = { gl: 0, glTD: 0, rzTgt: 0, rzTgtTD: 0, passAtt: 0, passTD: 0, rec: 0, recTD: 0, carries: 0, rushTD: 0, airY: 0, airTgt: 0, airTD: 0 };
    for (const s of seasons) for (const k in a) a[k] += league[s][k] || 0;
    return {
      GLCONV: a.gl ? a.glTD / a.gl : 0.45, RZTGTCONV: a.rzTgt ? a.rzTgtTD / a.rzTgt : 0.2, TGTTD: a.passAtt ? a.passTD / a.passAtt : 0.05,
      RECTD: a.rec ? a.recTD / a.rec : 0.08, RUSHTDATT: a.carries ? a.rushTD / a.carries : 0.03, AIRYDTD: a.airY ? a.airTD / a.airY : 0.0008,
    };
  }
  // live constants come from the two latest completed seasons; the backtest uses its train season only
  const liveSeasons = fullSeasons.filter(s => s >= CUR - 2);
  const C = constFor(liveSeasons.length ? liveSeasons : gotSeasons);
  // kappa: offensive TDs per point of closing total, over games with a line
  function kappaFor(seasons, filt = () => true) {
    let td = 0, pts = 0, n = 0;
    for (const g of games.values()) if (seasons.includes(g.s) && g.hasLine && filt(g)) {
      td += teamGames.get(g.gid + '|' + g.home).offTD + teamGames.get(g.gid + '|' + g.away).offTD; pts += g.total; n++; }
    return { k: pts ? td / pts : 0.12, n };
  }
  const kS = liveSeasons.length ? liveSeasons : gotSeasons;
  const KAPPA = kappaFor(kS).k;
  const both = g => g.fbsH && g.fbsA, oneFcs = g => !(g.fbsH && g.fbsA);
  const bigSpread = g => Math.abs(g.hMargin) >= 20, smallSpread = g => Math.abs(g.hMargin) < 10;
  log(`  kappa (off TDs per closing-total point, ${kS.join('+')}): ${KAPPA.toFixed(4)} | FBS-FBS ${kappaFor(kS, both).k.toFixed(4)} (n ${kappaFor(kS, both).n}) FBS-FCS ${kappaFor(kS, oneFcs).k.toFixed(4)} (n ${kappaFor(kS, oneFcs).n}) | spread<10 ${kappaFor(kS, smallSpread).k.toFixed(4)} spread>=20 ${kappaFor(kS, bigSpread).k.toFixed(4)}`);
  log(`  league constants: GLconv ${C.GLCONV.toFixed(3)} RZtgtTD ${C.RZTGTCONV.toFixed(3)} TD/pass att ${C.TGTTD.toFixed(3)} TD/catch ${C.RECTD.toFixed(3)} rushTD/carry ${C.RUSHTDATT.toFixed(4)} airYdTD ${C.AIRYDTD.toFixed(5)}`);
  for (const s of gotSeasons) { const L = league[s]; log(`    ${s}: carries ${L.carries} kneels ${L.kneel} named targets ${L.tgt} of ${L.passAtt} attempts (${pct(L.tgt, L.passAtt)}) catches ${L.rec}; air yards on ${pct(L.airTgt, L.tgt)} of targets`); }

  // ---- defensive / special-teams TD rates and giveaways, per FBS team-game ----
  function dstRates(seasons) {
    let n = 0, def = 0, ret = 0, st = 0, give = 0;
    for (const tg of teamGames.values()) if (seasons.includes(tg.s) && FBS[tg.s].has(tg.team)) { n++; def += tg.defTD; ret += tg.krTD + tg.prTD; st += tg.stTD; give += tg.give; }
    return { base: r4(def / n), stBase: r4(ret / n), stOther: r4(st / n), leagueGive: r4(give / n) };
  }
  const DST0 = dstRates(kS);
  log(`  per FBS team-game: defensive TDs ${DST0.base} | kick/punt return TDs ${DST0.stBase} | other special-teams TDs ${DST0.stOther} | giveaways ${DST0.leagueGive}`);

  // ---- per-team recency-weighted profiles ----
  const byTeam = new Map();   // team -> [team-game records]
  for (const tg of teamGames.values()) { let a = byTeam.get(tg.team); if (!a) { a = []; byTeam.set(tg.team, a); } a.push(tg); }
  for (const a of byTeam.values()) a.sort((x, y) => x.s - y.s || x.wk - y.wk);
  const leagueOffTDpg = (() => { let t = 0, n = 0; for (const tg of teamGames.values()) if (kS.includes(tg.s) && FBS[tg.s].has(tg.team)) { t += tg.offTD; n++; } return n ? t / n : 3.2; })();
  const GIVE_SHRINK = 8;   // pseudo-games of league-average giveaways in every team's rate
  const profiles = {};
  for (const [team, arr] of byTeam) {
    let wG = 0, wOff = 0, wRush = 0, wPass = 0, wPlays = 0, wGive = 0, wAllow = 0, wRA = 0, wPA = 0; const wBy = { RB: 0, WR: 0, TE: 0, QB: 0 };
    for (const tg of arr) {
      const w = SEASON_WEIGHT[tg.s] || 0; if (!w) continue;
      wG += w; wOff += w * tg.offTD; wRush += w * tg.rushTD; wPass += w * tg.passTD; wPlays += w * tg.plays; wGive += w * tg.give;
      wAllow += w * tg.tdAllow; wRA += w * tg.rushAllow; wPA += w * tg.passAllow; for (const k in wBy) wBy[k] += w * tg.byPos[k];
    }
    if (!wG) continue;
    const cur = arr.filter(tg => tg.s === CUR);
    let W = 0, Lx = 0, pf = 0, pa = 0;
    for (const tg of cur) { const g = games.get(tg.gid); const me = tg.home ? g.hs : g.as, op = tg.home ? g.as : g.hs; pf += me; pa += op; if (me > op) W++; else if (op > me) Lx++; }
    profiles[team] = {
      offTDpg: r3(wOff / wG), rushShare: r3((wRush + 0.5) / (wRush + wPass + 1)), playsPg: +(wPlays / wG).toFixed(1),
      giveawayPg: r3((wGive + GIVE_SHRINK * DST0.leagueGive) / (wG + GIVE_SHRINK)),
      gamesCur: cur.length, record: `${W}-${Lx}`, ppgCur: cur.length ? +(pf / cur.length).toFixed(1) : null, oppPpgCur: cur.length ? +(pa / cur.length).toFixed(1) : null,
      def: { tdAllowPg: r3(wAllow / wG), rushAllowShare: r3((wRA + 0.5) / (wRA + wPA + 1)), byPos: Object.fromEntries(Object.entries(wBy).map(([k, v]) => [k, r3(v / wG)])) },
    };
  }
  const leagueByPos = {};
  for (const k of ['RB', 'WR', 'TE', 'QB']) { let s = 0, n = 0; for (const [t, p] of Object.entries(profiles)) if (FBS[CUR].has(t)) { s += p.def.byPos[k]; n++; } leagueByPos[k] = r3(n ? s / n : 0); }

  // ---- per-player usage aggregation (recency-weighted) ----
  const AGG_KEYS = ['rushAtt', 'kneel', 'rushTD', 'glCarry', 'tgt', 'rec', 'rzTgt', 'recTD', 'airY', 'airTgt', 'kr', 'pr', 'stTD', 'drop'];
  const emptyAgg = () => { const a = { g: 0 }; for (const k of AGG_KEYS) a[k] = 0; return a; };
  const touchedRec = r => r.rushAtt > 0 || r.tgt > 0 || r.drop > 0;
  function addRec(a, r, w) { if (touchedRec(r)) a.g += w; for (const k of AGG_KEYS) a[k] += w * r[k]; }
  function aggregate(recs, weightOf) { const a = emptyAgg(); for (const r of recs) { const w = weightOf(r); if (w) addRec(a, r, w); } return a; }
  const liveW = r => SEASON_WEIGHT[r.s] || 0;
  const usageBucket = a => (a.drop > a.rushAtt && a.drop > 5 ? 'QB' : a.rushAtt > a.tgt ? 'RB' : a.tgt > 0 ? 'WR' : null);
  // one game's scoring weight for a player (the channels() formula applied to a single game) — used to MEASURE priors
  const gameWeight = (r, Cx, recVol) => ({
    rush: 0.40 * r.rushTD + 0.40 * r.glCarry * Cx.GLCONV + 0.20 * (r.rushAtt - r.kneel) * Cx.RUSHTDATT,
    rec: 0.35 * r.recTD + 0.35 * r.rzTgt * Cx.RZTGTCONV + 0.30 * (recVol === 'rec' ? r.rec * Cx.RECTD : r.tgt * Cx.TGTTD) });
  // New-player prior: in `season`, the average per-team-game production of rostered skill players who had NO
  // play-by-play history before that game (earlier seasons or earlier weeks). Most never touch the ball, which is the
  // point — this is what an unknown name on a college roster is worth.
  function newPriorFor(season, Cx, recVol) {
    const acc = {};
    for (const [team, ids] of Object.entries(seasonRoster[season] || {})) {
      const tgs = (byTeam.get(team) || []).filter(tg => tg.s === season); if (!tgs.length) continue;
      for (const pid of ids) {
        const bk = bucketFor(pid, season); if (!bk) continue;
        const recs = playerGames.get(pid) || [];
        if (recs.some(r => r.s < season)) continue;
        const firstWk = Math.min(Infinity, ...recs.filter(r => r.s === season && (touchedRec(r) || r.kr || r.pr)).map(r => r.wk));
        const a = acc[bk] || (acc[bk] = { n: 0, rush: 0, rec: 0 });
        for (const tg of tgs) {
          if (tg.wk > firstWk) break;                       // from his first touch on he has history
          const pg = pgIndex.get(pid + '|' + tg.gid); a.n++;
          if (pg && pg.team === team) { const w = gameWeight(pg, Cx, recVol); a.rush += w.rush; a.rec += w.rec; }
        }
      }
    }
    const out = {}; for (const [bk, a] of Object.entries(acc)) out[bk] = { rush: r4(a.rush / a.n), rec: r4(a.rec / a.n), n: a.n };
    return out;
  }
  // Presumed starting QB — ONE rule for the live roster and the backtest: last game's QB, then most games led, then
  // this season's dropbacks, then all-time (recency-weighted) dropbacks.
  function pickStarterQB(list) {
    const s = list.slice().sort((x, y) => ((y.isLast ? 1 : 0) - (x.isLast ? 1 : 0)) || (y.led - x.led) || (y.dropCur - x.dropCur) || (y.dropAll - x.dropAll));
    return s.length ? s[0].id : null;
  }
  // The QB of one team-game: ESPN's flagged starter (two flagged -> more 1st-half dropbacks), else the 1st-half
  // dropback leader. Garbage-time dropbacks never make a backup "the QB".
  function gameQBof(gid, team, isQB) {
    const st = starters.get(gid), recs = (tgPlayers.get(gid + '|' + team) || []).filter(r => isQB(r.pid));
    const flagged = st ? recs.filter(r => st.has(r.pid)) : [];
    if (flagged.length) return flagged.reduce((a, b) => (b.drop1h > a.drop1h ? b : a)).pid;
    let best = null, bd = 4; for (const r of recs) if (r.drop1h > bd) { bd = r.drop1h; best = r.pid; }
    return best;
  }

  // ==========================================================================
  // BACKTEST (rolling, leak-free). For each TEST_SEASON week W, every game is predicted with ONLY: player history from
  // earlier seasons (weighted wPrior^age), TEST_SEASON weeks < W, and constants / kappa / priors from the TRAIN season;
  // anchored to that game's real closing total + spread. It runs the app's exact math (cfb-model.js distribute()).
  // Candidate sets: 'touched' = players who touched the ball (or returned a kick) in that game — like knowing the
  // actives, comparable to the NFL headline; 'roster' = + everyone seen for the team earlier in the season;
  // 'full' = + the team's whole skill-position roster — what the app actually prices (no injury feed in college).
  // ==========================================================================
  const btOut = { backtest: null, dstBacktest: null, roleCal: {}, dstFit: { slope: 0.06, giveExp: 1 }, experiments: null };
  const summarize = (rows, pk = 'p', yk = 'y') => {
    let n = 0, b = 0, ll = 0, pos = 0, sp = 0; const bins = Array.from({ length: 10 }, () => ({ n: 0, y: 0, p: 0 }));
    for (const r of rows) { const p = Math.min(1 - 1e-4, Math.max(1e-4, r[pk])), y = r[yk]; n++; pos += y; sp += p; b += (p - y) ** 2; ll += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      const bb = bins[Math.min(9, Math.floor(p * 10))]; bb.n++; bb.y += y; bb.p += p; }
    const base = n ? pos / n : 0;
    return { n, brier: n ? b / n : null, logloss: n ? ll / n : null, baseRate: base, baselineBrier: base * (1 - base), skill: n ? 1 - (b / n) / (base * (1 - base)) : null,
      meanPred: n ? sp / n : null, reliability: bins.filter(x => x.n >= 25).map(x => ({ pred: r3(x.p / x.n), actual: r3(x.y / x.n), n: x.n })) };
  };
  const groupCal = (rows, keyFn, pk = 'p', yk = 'y') => {
    const t = {}; for (const r of rows) { const k = keyFn(r); if (k == null) continue; const a = t[k] || (t[k] = { n: 0, p: 0, y: 0 }); a.n++; a.p += r[pk]; a.y += r[yk]; }
    return Object.entries(t).map(([k, a]) => { const pr = a.p / a.n, ac = a.y / a.n, se = Math.sqrt(Math.max(ac * (1 - ac), 1e-6) / a.n);
      return { k, n: a.n, pred: r3(pr), actual: r3(ac), gapSE: +((ac - pr) / se).toFixed(1) }; });
  };
  const fmtCal = g => g.map(x => `${x.k} ${(x.pred * 100).toFixed(1)}->${(x.actual * 100).toFixed(1)}% (n ${x.n}${Math.abs(x.gapSE) >= 2 ? `, ${x.gapSE > 0 ? '+' : ''}${x.gapSE} SE` : ''})`).join(' | ');
  const H1 = [1, 7], H2 = [8, 99];
  const inH = (r, h) => r.wk >= h[0] && r.wk <= h[1];
  const brierOf = rows => rows.reduce((s, r) => s + (r.p - r.y) ** 2, 0) / (rows.length || 1);
  const marginBucket = m => (m <= -21 ? 'dog 21+' : m <= -10 ? 'dog 10-21' : m < 0 ? 'dog <10' : m < 10 ? 'fav <10' : m < 21 ? 'fav 10-21' : 'fav 21+');

  if (TEST_SEASON && TRAIN_SEASON && gotSeasons.includes(TEST_SEASON) && gotSeasons.includes(TRAIN_SEASON)) {
    log(`\n  --- backtest: train ${TRAIN_SEASON}, test every ${TEST_SEASON} game with an FBS team (closing lines, leak-free) ---`);
    const histWx = await loadHistWeather(TEST_SEASON);
    const testGames = [...games.values()].filter(g => g.s === TEST_SEASON);
    const WEEKS = [...new Set(testGames.map(g => g.wk))].sort((a, b) => a - b);
    const byWeek = new Map(); for (const g of testGames) { if (!byWeek.has(g.wk)) byWeek.set(g.wk, []); byWeek.get(g.wk).push(g); }
    const Ctrain = constFor([TRAIN_SEASON]);
    const Ktrain = kappaFor([TRAIN_SEASON]).k;
    const DSTtrain = dstRates([TRAIN_SEASON]);
    const npCache = {}; const npFor = rv => npCache[rv] || (npCache[rv] = newPriorFor(TRAIN_SEASON, Ctrain, rv));
    { let a = 0, b = 0; for (const g of games.values()) if (g.s === TRAIN_SEASON && g.hasLine) for (const side of ['home', 'away']) { const m = CFBModel.expOffTD(g.total, side === 'home' ? g.hMargin : -g.hMargin, Ktrain, MODEL), y = teamGames.get(g.gid + '|' + g[side]).offTD; a += m * m; b += (y - m) ** 2 - m; }
      log(`  team offensive-TD overdispersion on ${TRAIN_SEASON}: NB shape ~ ${(b > 0 ? a / b : Infinity).toFixed(1)} (larger = closer to Poisson; MODEL uses ${MODEL.nbSize})`); }
    log(`  train constants: kappa ${Ktrain.toFixed(4)}, new-player prior (per team-game) ${Object.entries(npFor('tgt')).map(([k, v]) => `${k} rush ${v.rush} rec ${v.rec} (n ${v.n})`).join('; ')}`);
    const histCache = {};
    function histFor(wP) {   // everything before the test season, recency-weighted
      if (histCache[wP]) return histCache[wP];
      const pl = new Map(), tm = new Map();
      for (const [pid, recs] of playerGames) { let a = null; for (const r of recs) if (r.s < TEST_SEASON) { a = a || emptyAgg(); addRec(a, r, Math.pow(wP, TEST_SEASON - r.s)); } if (a) pl.set(pid, a); }
      for (const tg of teamGames.values()) if (tg.s < TEST_SEASON) { const w = Math.pow(wP, TEST_SEASON - tg.s); const t = tm.get(tg.team) || { rushTD: 0, passTD: 0 }; t.rushTD += w * tg.rushTD; t.passTD += w * tg.passTD; tm.set(tg.team, t); }
      return (histCache[wP] = { pl, tm });
    }
    const sumAgg = (a, b) => { const o = emptyAgg(); o.g = a.g + b.g; for (const k of AGG_KEYS) o[k] = a[k] + b[k]; return o; };
    const isQBtest = pid => bucketFor(pid, TEST_SEASON) === 'QB';
    const touchedIds = key => (tgPlayers.get(key) || []).filter(r => touchedRec(r) || r.kr > 0 || r.pr > 0).map(r => r.pid);

    function runBacktest(o) {
      const M = { ...MODEL, ...o };
      const Cb = { ...Ctrain, NEWPRIOR: npFor(M.recVol) };
      const H = histFor(M.wPrior);
      const cur = new Map(), teamCur = new Map(), seen = new Map(), lastIdx = new Map(), qbState = new Map();
      const rows = [];
      for (const wk of WEEKS) {
        const scoreWk = !o.wk || (wk >= o.wk[0] && wk <= o.wk[1]);   // all weeks still fold; only the window is scored
        if (scoreWk) for (const g of byWeek.get(wk)) {
          if (!g.hasLine) continue;
          const wx = histWx.get(g.gid) || null, sides = [];
          for (const side of ['home', 'away']) {
            const team = g[side], margin = side === 'home' ? g.hMargin : -g.hMargin;
            const ids = new Set(touchedIds(g.gid + '|' + team));
            if (o.cand !== 'touched') for (const pid of seen.get(team) || []) ids.add(pid);
            if (o.cand === 'full') for (const pid of (seasonRoster[TEST_SEASON] || {})[team] || []) ids.add(pid);
            const tc = teamCur.get(team) || { rushTD: 0, passTD: 0, n: 0 }, ht = H.tm.get(team) || { rushTD: 0, passTD: 0 };
            const rushBase = (ht.rushTD + tc.rushTD + 0.5) / (ht.rushTD + ht.passTD + tc.rushTD + tc.passTD + 1);
            const cands = [];
            for (const pid of ids) {
              const h = H.pl.get(pid), c = cur.get(pid), ag = h && c ? sumAgg(h, c) : (h || c || null);
              const bk = bucketFor(pid, TEST_SEASON) || (ag && usageBucket(ag)); if (!bk) continue;
              const ch = CFBModel.channels(ag, bk, M, Cb), li = lastIdx.get(pid + '|' + team);
              cands.push({ pid, bk, rush: ch.rush, rec: ch.rec, ret: ag ? ag.kr + ag.pr : 0, gamesAgo: li == null ? null : tc.n - 1 - li,
                touchPg: ag && ag.g ? (ag.rushAtt - ag.kneel + ag.tgt) / ag.g : 0, hasHist: !!(ag && ag.g > 0), qbStarter: null,
                dropCur: c ? c.drop : 0, dropAll: ag ? ag.drop : 0 });
            }
            const qs = qbState.get(team), qbs = cands.filter(c => c.bk === 'QB');
            const st = pickStarterQB(qbs.map(c => ({ id: c.pid, isLast: !!qs && qs.last === c.pid, led: qs ? qs.led.get(c.pid) || 0 : 0, dropCur: c.dropCur, dropAll: c.dropAll })));
            for (const c of qbs) c.qbStarter = c.pid === st;
            const res = CFBModel.distribute(cands, { total: g.total, margin, rushBase, wx, teamGames: tc.n, K: Ktrain, stBase: DSTtrain.stBase }, M);
            sides.push({ team, margin, cands, res });
          }
          // first-TD diagnostic: P(player scores the game's first TD) ~ P(any TD) x his share of the game's TD rate
          let E = 2 * (DSTtrain.base + DSTtrain.stOther), P0 = Math.exp(-E);
          for (const s of sides) { const teo = s.res.reduce((a, x) => a + x.expOff, 0), tst = s.res.reduce((a, x) => a + x.expST, 0);
            E += CFBModel.expOffTD(g.total, s.margin, Ktrain, M) + DSTtrain.stBase; P0 *= CFBModel.nbP0(teo, M.nbSize) * Math.exp(-tst); }
          const first = g.tds[0] || null;
          for (const s of sides) for (let i = 0; i < s.cands.length; i++) {
            const c = s.cands[i], x = s.res[i], pg = pgIndex.get(c.pid + '|' + g.gid);
            const n = pg && pg.team === s.team ? pg.rushTD + pg.recTD + pg.stTD : 0;
            rows.push({ gid: g.gid, pid: c.pid, team: s.team, wk, p: x.p, p2: x.p2, y: n > 0 ? 1 : 0, y2: n >= 2 ? 1 : 0, bk: c.bk, tier: x.tier,
              margin: s.margin, fcs: !FBS[TEST_SEASON].has(s.team), hist: c.hasHist, qbS: c.qbStarter, ago: c.gamesAgo,
              pF: (1 - P0) * (x.expOff + x.expST) / E, yF: first && first.pid === c.pid && first.team === s.team ? 1 : 0 });
          }
        }
        // fold week wk: only now does it become visible to later weeks
        for (const g of byWeek.get(wk)) for (const team of [g.home, g.away]) {
          const tg = teamGames.get(g.gid + '|' + team), tc = teamCur.get(team) || { rushTD: 0, passTD: 0, n: 0 };
          tc.rushTD += tg.rushTD; tc.passTD += tg.passTD; tc.n++; teamCur.set(team, tc);
          for (const r of tgPlayers.get(g.gid + '|' + team) || []) {
            let a = cur.get(r.pid); if (!a) { a = emptyAgg(); cur.set(r.pid, a); } addRec(a, r, 1);
            if (touchedRec(r) || r.kr > 0 || r.pr > 0) { let s = seen.get(team); if (!s) { s = new Set(); seen.set(team, s); } s.add(r.pid); lastIdx.set(r.pid + '|' + team, tc.n - 1); }
          }
          const gq = gameQBof(g.gid, team, isQBtest); let qs = qbState.get(team); if (!qs) { qs = { last: null, led: new Map() }; qbState.set(team, qs); }
          qs.last = gq; if (gq) qs.led.set(gq, (qs.led.get(gq) || 0) + 1);
        }
      }
      return rows;
    }

    // ---- role (usage-tier) recalibration, cross-fitted: factors learned on one half, applied to the other ----
    const TIERS = ['qb', 'feature', 'regular', 'rotation', 'fringe', 'new'];
    const tierFactors = rows => { const f = {}; for (const g of groupCal(rows, r => r.tier)) f[g.k] = g.n >= 150 && g.pred > 0 ? +Math.max(0.7, Math.min(1.3, g.actual / g.pred)).toFixed(3) : 1; return f; };
    const applyCal = (rows, f) => rows.map(r => ({ ...r, p: CFBModel.roleCal(r.p, r.tier, f), pF: Math.min(0.97, r.pF * (f[r.tier] || 1)) }));
    function crossFit(rows, keep) {   // keep = tiers allowed to be recalibrated
      const a = rows.filter(r => inH(r, H1)), b = rows.filter(r => inH(r, H2));
      const pick = f => Object.fromEntries(Object.entries(f).filter(([k]) => keep.includes(k)));
      const fa = pick(tierFactors(a)), fb = pick(tierFactors(b));
      return { fa, fb, h1Raw: brierOf(a), h1Cal: brierOf(applyCal(a, fb)), h2Raw: brierOf(b), h2Cal: brierOf(applyCal(b, fa)) };
    }

    if (process.env.BT_EXPERIMENTS) {
      // Validation harness: flip each switch; score weeks 1-7 (H1) and 8+ (H2) separately. A change earns its place only
      // if it helps on BOTH halves. Distribution switches are judged on 'touched' (who played is known); availability and
      // new-player switches on 'full' (their job is handling players who may not play — invisible in 'touched').
      const base = { ...MODEL };
      const bri = (o, cand, h) => brierOf(runBacktest({ ...o, cand, wk: h }));
      // final confirmation pass: neighbours of every shipped value (none should win on both halves of the set it's judged on)
      const flips = [['qbRush', 1.2], ['qbRush', 1.6], ['qbBackup', 0.15], ['qbBackup', 0.45], ['kSlope', 0], ['kSlope', 0.1], ['nbSize', 4], ['nbSize', 6],
        ['aNone', 0.3], ['a1', 0.6], ['avail', false], ['newPrior', 'fixed'], ['eps', 0.005], ['eps', 0.02], ['gt', 0.2], ['weather', true], ['wPrior', 0.5], ['recVol', 'rec'], ['air', true]];
      const sg = d => (d >= 0 ? '+' : '') + d.toFixed(5);
      const MIN_GAIN = 0.00003;   // a flip must beat the current model by at least this on each half (smaller = noise)
      const baseB = {}; for (const cand of ['touched', 'full']) baseB[cand] = [bri(base, cand, H1), bri(base, cand, H2)];
      log(`  --- MODEL switch check (current: touched H1 ${baseB.touched[0].toFixed(5)} H2 ${baseB.touched[1].toFixed(5)} | full H1 ${baseB.full[0].toFixed(5)} H2 ${baseB.full[1].toFixed(5)}; negative = the flip is better; <== = better by ${MIN_GAIN}+ on both halves) ---`);
      const exp = [];
      for (const [k, v] of flips) {
        const o = { ...base, [k]: v }, e = { k, v };
        for (const cand of ['touched', 'full']) { e[cand] = [bri(o, cand, H1) - baseB[cand][0], bri(o, cand, H2) - baseB[cand][1]]; }
        const win = c => e[c][0] <= -MIN_GAIN && e[c][1] <= -MIN_GAIN ? '<==' : '   ';
        exp.push(e);
        log(`  ${(k + ' -> ' + v).padEnd(22)} touched H1 ${sg(e.touched[0])} H2 ${sg(e.touched[1])} ${win('touched')} | full H1 ${sg(e.full[0])} H2 ${sg(e.full[1])} ${win('full')}`);
      }
      btOut.experiments = exp.map(e => ({ k: e.k, v: e.v, touched: e.touched.map(x => +x.toFixed(6)), full: e.full.map(x => +x.toFixed(6)) }));
      const rt = runBacktest({ ...base, cand: 'touched' }), rf = runBacktest({ ...base, cand: 'full' });
      log(`  by position [touched]: ${fmtCal(groupCal(rt, r => r.bk))}`);
      log(`  by usage tier [touched]: ${fmtCal(groupCal(rt, r => r.tier))}`);
      log(`  by usage tier [full]: ${fmtCal(groupCal(rf, r => r.tier))}`);
      log(`  QB starter vs backup [full]: ${fmtCal(groupCal(rf.filter(r => r.bk === 'QB'), r => (r.qbS ? 'starter' : 'backup')))}`);
      log(`  by spread [touched]: ${fmtCal(groupCal(rt, r => marginBucket(r.margin)))}`);
      log(`  feature/regular by spread [touched]: ${fmtCal(groupCal(rt.filter(r => r.tier === 'feature' || r.tier === 'regular'), r => marginBucket(r.margin)))}`);
      log(`  rotation/fringe/new by spread [touched]: ${fmtCal(groupCal(rt.filter(r => ['rotation', 'fringe', 'new'].includes(r.tier)), r => marginBucket(r.margin)))}`);
      log(`  availability (team games since last touch) [full]: ${fmtCal(groupCal(rf, r => (r.ago == null ? 'none yet' : r.ago >= 2 ? '2+' : String(r.ago))))}`);
      log(`  FCS side vs FBS side [touched]: ${fmtCal(groupCal(rt, r => (r.fcs ? 'FCS team' : 'FBS team')))}`);
      for (const [cand, rows] of [['touched', rt], ['full', rf]]) {
        const a = rows.filter(r => inH(r, H1)), b = rows.filter(r => inH(r, H2)), fac = x => x.reduce((s, r) => s + r.y2, 0) / x.reduce((s, r) => s + r.p2, 0);
        const b2 = (x, f) => x.reduce((s, r) => s + (Math.min(r.p, r.p2 * f) - r.y2) ** 2, 0) / x.length;
        log(`  2+ TD cross-fit [${cand}]: factor H1 ${fac(a).toFixed(3)} H2 ${fac(b).toFixed(3)}; H1 ${b2(a, 1).toFixed(5)} -> ${b2(a, fac(b)).toFixed(5)}, H2 ${b2(b, 1).toFixed(5)} -> ${b2(b, fac(a)).toFixed(5)}`);
      }
      for (const cand of ['touched', 'full']) {
        const rows = cand === 'touched' ? rt : rf, all = crossFit(rows, TIERS);
        log(`  role recalibration cross-fit [${cand}]: H1 ${all.h1Raw.toFixed(5)} -> ${all.h1Cal.toFixed(5)} (factors from H2 ${JSON.stringify(all.fb)}) | H2 ${all.h2Raw.toFixed(5)} -> ${all.h2Cal.toFixed(5)} (factors from H1 ${JSON.stringify(all.fa)})`);
        for (const t of TIERS) { const one = crossFit(rows, [t]); log(`    only ${t.padEnd(8)}: H1 ${sg(one.h1Cal - one.h1Raw)} H2 ${sg(one.h2Cal - one.h2Raw)}${one.h1Cal < one.h1Raw && one.h2Cal < one.h2Raw ? '  <-- helps both' : ''}  (H1 f ${one.fb[t] ?? '-'}, H2 f ${one.fa[t] ?? '-'})`); }
      }
    }

    // ---- headline numbers for the shipped MODEL ----
    const rowsT = runBacktest({ ...MODEL, cand: 'touched' }), rowsF = runBacktest({ ...MODEL, cand: 'full' });
    // role recalibration: only tiers listed in MODEL.roleCalTiers ship (chosen by the cross-fit above); the headline
    // is scored honestly — each half recalibrated with factors learned on the OTHER half.
    const calTiers = MODEL.roleCalTiers || [];
    const shipCal = rows => { const f = tierFactors(rows); return Object.fromEntries(calTiers.filter(t => f[t] != null).map(t => [t, f[t]])); };
    const crossCal = rows => { const cf = crossFit(rows, calTiers); return [...applyCal(rows.filter(r => inH(r, H1)), cf.fb), ...applyCal(rows.filter(r => inH(r, H2)), cf.fa)]; };
    const calT = crossCal(rowsT), calF = crossCal(rowsF);
    btOut.roleCal = shipCal(rowsF);   // the app prices the whole roster, so its factors come from the 'full' set
    const sT = summarize(calT), sF = summarize(calF), s2 = summarize(calT, 'p2', 'y2'), sFirst = summarize(calT, 'pF', 'yF');
    const pctS = x => (x * 100).toFixed(1) + '%';
    log(`  headline [touched — actives known, like the NFL headline]: N ${sT.n} Brier ${sT.brier.toFixed(4)} (base ${sT.baselineBrier.toFixed(4)}, skill ${pctS(sT.skill)}) logloss ${sT.logloss.toFixed(4)} meanPred ${pctS(sT.meanPred)} base ${pctS(sT.baseRate)}`);
    log(`    halves: H1 ${brierOf(calT.filter(r => inH(r, H1))).toFixed(4)} H2 ${brierOf(calT.filter(r => inH(r, H2))).toFixed(4)}; reliability ${sT.reliability.map(b => `${(b.pred * 100) | 0}->${(b.actual * 100) | 0}%(${b.n})`).join(' ')}`);
    log(`  whole roster [full — what the app prices]: N ${sF.n} Brier ${sF.brier.toFixed(4)} (base ${sF.baselineBrier.toFixed(4)}, skill ${pctS(sF.skill)}) logloss ${sF.logloss.toFixed(4)} meanPred ${pctS(sF.meanPred)} base ${pctS(sF.baseRate)}`);
    log(`    reliability ${sF.reliability.map(b => `${(b.pred * 100) | 0}->${(b.actual * 100) | 0}%(${b.n})`).join(' ')}`);
    log(`  2+ TDs [touched]: Brier ${s2.brier.toFixed(4)} (base ${s2.baselineBrier.toFixed(4)}, skill ${pctS(s2.skill)}); reliability ${s2.reliability.map(b => `${(b.pred * 100) | 0}->${(b.actual * 100) | 0}%(${b.n})`).join(' ')}`);
    log(`  1st TD [touched]: Brier ${sFirst.brier.toFixed(5)} (base ${sFirst.baselineBrier.toFixed(5)}, skill ${pctS(sFirst.skill)}); mean pred ${pctS(sFirst.meanPred)} vs actual ${pctS(sFirst.baseRate)}`);
    log(`  by position [touched]: ${fmtCal(groupCal(calT, r => r.bk))}`);
    log(`  by usage tier [full]: ${fmtCal(groupCal(calF, r => r.tier))}`);
    log(`  role recalibration shipped: ${JSON.stringify(btOut.roleCal)} (tiers ${calTiers.join(',') || 'none'})`);
    const pack = (s, rows) => ({ n: s.n, brier: r4(s.brier), logloss: r4(s.logloss), baselineBrier: r4(s.baselineBrier), skill: r4(s.skill), baseRate: r4(s.baseRate), meanPred: r4(s.meanPred),
      h1: r4(brierOf(rows.filter(r => inH(r, H1)))), h2: r4(brierOf(rows.filter(r => inH(r, H2)))), reliability: s.reliability });
    btOut.backtest = {
      trainSeason: TRAIN_SEASON, testSeason: TEST_SEASON, games: testGames.filter(g => g.hasLine).length, method: 'rolling within-season, real closing lines',
      touched: pack(sT, calT), full: pack(sF, calF), twoPlus: pack(s2, calT.map(r => ({ ...r, p: r.p2, y: r.y2 }))),
      firstTD: { brier: +sFirst.brier.toFixed(5), baselineBrier: +sFirst.baselineBrier.toFixed(5), meanPred: r4(sFirst.meanPred), actual: r4(sFirst.baseRate) },
      byPosition: groupCal(calT, r => r.bk), byTier: groupCal(calF, r => r.tier), bySpread: groupCal(calT, r => marginBucket(r.margin)),
      kappa: r4(Ktrain), roleCal: btOut.roleCal, model: MODEL, experiments: btOut.experiments,
    };

    // ---- market-universe check: only the players ESPN BET listed an anytime-TD line for (lines, not prices) ----
    const mlPath = path.join(__dirname, `market_lines_${TEST_SEASON}.json`);
    if (fs.existsSync(mlPath)) {
      const ml = JSON.parse(fs.readFileSync(mlPath, 'utf8')).games || {};
      const listed = {}; for (const [gid, x] of Object.entries(ml)) if ((x.market || []).length) listed[gid] = new Set(x.market);
      const inMk = calT.filter(r => listed[r.gid] && listed[r.gid].has(r.pid));
      const sc = calT.filter(r => listed[r.gid] && r.y), scL = sc.filter(r => listed[r.gid].has(r.pid));
      const sM = summarize(inMk);
      btOut.backtest.market = { games: Object.keys(listed).length, n: sM.n, brier: r4(sM.brier), baselineBrier: r4(sM.baselineBrier), skill: r4(sM.skill), meanPred: r4(sM.meanPred), baseRate: r4(sM.baseRate),
        scorerRecall: r4(scL.length / (sc.length || 1)), avgListed: +(Object.values(listed).reduce((a, s) => a + s.size, 0) / (Object.keys(listed).length || 1)).toFixed(1), reliability: sM.reliability };
      log(`  market universe (ESPN BET anytime-TD boards, ${btOut.backtest.market.games} games, ~${btOut.backtest.market.avgListed} players listed/game): N ${sM.n} Brier ${sM.brier.toFixed(4)} (base ${sM.baselineBrier.toFixed(4)}, skill ${pctS(sM.skill)}); book listed ${pctS(scL.length / (sc.length || 1))} of actual scorers`);
    } else log(`  (no ${path.basename(mlPath)} — run build-market-lines.mjs to add the market-universe check)`);

    // ---- defense-TD model: fit slope / giveaway exponent on the TRAIN season, test on the TEST season (held out) ----
    function dstRows(season) {   // rolling as-of opponent giveaway rate: earlier seasons (wPrior^age) + this season's weeks < W
      const hist = new Map();
      for (const tg of teamGames.values()) if (tg.s < season) { const w = Math.pow(MODEL.wPrior, season - tg.s); const t = hist.get(tg.team) || { g: 0, give: 0 }; t.g += w; t.give += w * tg.give; hist.set(tg.team, t); }
      const lg = dstRates([season - 1].filter(s => gotSeasons.includes(s))).leagueGive || DST0.leagueGive;
      const run = new Map(), out = [], sg = [...games.values()].filter(g => g.s === season);
      for (const wk of [...new Set(sg.map(g => g.wk))].sort((a, b) => a - b)) {
        const gs = sg.filter(g => g.wk === wk);
        for (const g of gs) if (g.hasLine) for (const side of ['home', 'away']) {
          const me = g[side], opp = side === 'home' ? g.away : g.home, h = hist.get(opp) || { g: 0, give: 0 }, r = run.get(opp) || { g: 0, give: 0 };
          out.push({ margin: side === 'home' ? g.hMargin : -g.hMargin, oppGive: (h.give + r.give + GIVE_SHRINK * lg) / (h.g + r.g + GIVE_SHRINK), lg, y: teamGames.get(g.gid + '|' + me).defTD, wk });
        }
        for (const g of gs) for (const t of [g.home, g.away]) { const tg = teamGames.get(g.gid + '|' + t), r = run.get(t) || { g: 0, give: 0 }; r.g++; r.give += tg.give; run.set(t, r); }
      }
      return out;
    }
    const trainD = dstRows(TRAIN_SEASON), testD = dstRows(TEST_SEASON);
    const lam = (x, D) => D.base * Math.exp(D.slope * x.margin) * Math.pow(x.oppGive / x.lg, D.giveExp);
    const fitBase = (rows, slope, giveExp) => rows.reduce((s, x) => s + x.y, 0) / rows.reduce((s, x) => s + Math.exp(slope * x.margin) * Math.pow(x.oppGive / x.lg, giveExp), 0);
    let best = null;
    for (const slope of [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]) for (const giveExp of [0, 0.5, 1, 1.5]) {
      const D = { base: fitBase(trainD, slope, giveExp), slope, giveExp };
      const ll = trainD.reduce((s, x) => { const l = lam(x, D); return s + x.y * Math.log(l) - l; }, 0);
      if (!best || ll > best.ll) best = { ...D, ll };
    }
    const trainBase = trainD.reduce((s, x) => s + x.y, 0) / trainD.length;
    const scoreD = (rows, D) => rows.reduce((s, x) => s + ((1 - Math.exp(-lam(x, D))) - (x.y > 0 ? 1 : 0)) ** 2, 0) / rows.length;
    const Dfit = { base: best.base, slope: best.slope, giveExp: best.giveExp }, Dflat = { base: trainBase, slope: 0, giveExp: 0 }, Dnfl = { base: fitBase(trainD, 0.06, 1), slope: 0.06, giveExp: 1 };
    const half = (rows, h) => rows.filter(x => x.wk >= h[0] && x.wk <= h[1]);
    const pts = testD.map(x => ({ p: 1 - Math.exp(-lam(x, Dfit)), y: x.y > 0 ? 1 : 0 })).sort((a, b) => a.p - b.p), n3 = Math.floor(pts.length / 3);
    const third = sl => ({ pred: r3(sl.reduce((t, x) => t + x.p, 0) / sl.length), actual: r3(sl.reduce((t, x) => t + x.y, 0) / sl.length), n: sl.length });
    const liveRows = [TEST_SEASON, TRAIN_SEASON].flatMap(dstRows);
    btOut.dstFit = { slope: best.slope, giveExp: best.giveExp, base: r4(fitBase(liveRows, best.slope, best.giveExp)) };
    btOut.dstBacktest = { fitSeason: TRAIN_SEASON, testSeason: TEST_SEASON, n: testD.length, rate: r3(testD.filter(x => x.y > 0).length / testD.length), slope: best.slope, giveExp: best.giveExp,
      brier: r4(scoreD(testD, Dfit)), baselineBrier: r4(scoreD(testD, Dflat)), nflParamsBrier: r4(scoreD(testD, Dnfl)),
      h1: [r4(scoreD(half(testD, H1), Dfit)), r4(scoreD(half(testD, H1), Dflat))], h2: [r4(scoreD(half(testD, H2), Dfit)), r4(scoreD(half(testD, H2), Dflat))],
      thirds: [third(pts.slice(0, n3)), third(pts.slice(n3, 2 * n3)), third(pts.slice(2 * n3))] };
    const db = btOut.dstBacktest;
    log(`  defense TDs: fit on ${TRAIN_SEASON} -> slope ${best.slope}, giveaway exp ${best.giveExp}; tested on ${TEST_SEASON} (N ${db.n}, rate ${(db.rate * 100).toFixed(1)}%): Brier ${db.brier} vs league-average ${db.baselineBrier} vs NFL params ${db.nflParamsBrier}; H1 ${db.h1[0]} vs ${db.h1[1]}, H2 ${db.h2[0]} vs ${db.h2[1]}; by third (pred->actual) ${db.thirds.map(t => `${(t.pred * 100).toFixed(1)}->${(t.actual * 100).toFixed(1)}%`).join(' ')}`);
  } else log('  (backtest skipped: needs the train and test seasons in the season list)');
  const DST_MODEL = { ...DST0, base: btOut.dstFit.base || DST0.base, slope: btOut.dstFit.slope, giveExp: btOut.dstFit.giveExp, shrinkGames: GIVE_SHRINK };
  log(`  defense model shipped: ${JSON.stringify(DST_MODEL)}`);

  // ==========================================================================
  // ESPN live: slate, ranks, teams, rosters, venues
  // ==========================================================================
  log(`  pulling ESPN slate / rankings / teams / rosters ...`);
  const sb0 = await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300`);
  if (!sb0) throw new Error('ESPN scoreboard unavailable');
  // this week = the calendar week containing now; if every game in it is final, move to the next week
  const cal = (sb0.leagues && sb0.leagues[0] && sb0.leagues[0].calendar) || [];
  const weeksCal = [];
  for (const c of cal) for (const e of c.entries || []) weeksCal.push({ type: c.value, week: e.value, label: e.label, start: Date.parse(e.startDate), end: Date.parse(e.endDate) });
  const now = Date.now();
  let wi = weeksCal.findIndex(w => now >= w.start && now <= w.end); if (wi < 0) wi = weeksCal.findIndex(w => w.start > now);
  let sb = sb0, weekInfo = weeksCal[wi] || null;
  const allFinal = s => (s.events || []).length && s.events.every(e => e.status && e.status.type && e.status.type.completed);
  if (weekInfo) {
    sb = (await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&week=${weekInfo.week}&seasontype=${weekInfo.type}`)) || sb0;
    if (allFinal(sb) && weeksCal[wi + 1]) { weekInfo = weeksCal[wi + 1]; sb = (await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&week=${weekInfo.week}&seasontype=${weekInfo.type}`)) || sb; }
  }
  const schedule = (sb.events || []).map(e => {
    const c = e.competitions[0];
    const h = c.competitors.find(x => x.homeAway === 'home'), a = c.competitors.find(x => x.homeAway === 'away');
    const o = (c.odds || [])[0] || {};
    const total = o.overUnder != null && +o.overUnder > 20 ? +o.overUnder : null;
    let fav = null, spread = null;
    if (o.homeTeamOdds && o.homeTeamOdds.favorite) fav = h.team.id; else if (o.awayTeamOdds && o.awayTeamOdds.favorite) fav = a.team.id;
    if (o.spread != null) spread = Math.abs(+o.spread);
    if (!fav && o.details) { const m = o.details.match(/^([A-Z&]{2,6})\s*(-?\d+(?:\.\d)?)/); if (m) { fav = m[1] === h.team.abbreviation ? h.team.id : m[1] === a.team.abbreviation ? a.team.id : null; spread = Math.abs(+m[2]); } }
    if (o.details && /EVEN|PK/i.test(o.details)) { spread = 0; fav = null; }
    const rk = x => (x.curatedRank && x.curatedRank.current <= 25 ? x.curatedRank.current : null);
    const v = c.venue || {};
    return { id: e.id, home: h.team.id, away: a.team.id, kickoff: e.date, completed: !!(e.status && e.status.type && e.status.type.completed),
      neutral: !!c.neutralSite, confGame: !!c.conferenceCompetition, total, fav, spread, homeRank: rk(h), awayRank: rk(a),
      venue: { id: v.id || null, name: v.fullName || '', city: (v.address && v.address.city) || '', state: (v.address && v.address.state) || '', country: (v.address && v.address.country) || 'USA', indoor: !!v.indoor },
      tv: ((c.broadcasts || [])[0] && c.broadcasts[0].names || []).join('/') };
  });
  log(`  slate: ${weekInfo ? weekInfo.label : 'current'} — ${schedule.length} games, ${schedule.filter(g => g.total).length} with a line`);

  // rankings: AP Top 25 (+ CFP when it exists)
  const ranks = {};
  const rk = await fetchJson(`${ESPN}/rankings`);
  for (const poll of (rk && rk.rankings) || []) {
    const key = /AP Top 25/i.test(poll.name) ? 'ap' : /Playoff/i.test(poll.name) ? 'cfp' : null; if (!key) continue;
    for (const r of poll.ranks || []) { const id = r.team && r.team.id; if (!id) continue; (ranks[id] = ranks[id] || {})[key] = r.current; }
  }

  // teams to ship: every current FBS team + any non-FBS opponent on this week's slate
  const teamIds = new Set(FBS[CUR]);
  for (const g of schedule) { teamIds.add(g.home); teamIds.add(g.away); }
  const teamMeta = {};
  const espnTeams = await fetchJson(`${ESPN}/teams?limit=1000`);
  const allAbbr = {};   // every ESPN college team id -> abbreviation (names a transfer's previous school)
  for (const t of ((espnTeams && espnTeams.sports[0].leagues[0].teams) || []).map(x => x.team)) { allAbbr[t.id] = t.abbreviation; if (teamIds.has(t.id)) teamMeta[t.id] = t; }
  await pool([...teamIds].filter(id => !teamMeta[id]), 8, async id => { const j = await fetchJson(`${ESPN}/teams/${id}`); if (j && j.team) teamMeta[id] = j.team; });

  const teamsOut = {};
  await pool([...teamIds], 8, async id => {
    const t = teamMeta[id] || { id, abbreviation: id, displayName: id, shortDisplayName: id };
    let logo = null;
    if (!process.env.SKIP_LOGOS) {
      const b64 = curlBase64(`https://a.espncdn.com/combiner/i?img=/i/teamlogos/ncaa/500/${id}.png&w=64&h=64`);
      if (b64) logo = 'data:image/png;base64,' + b64;
    }
    const conf = CONF_OF[CUR][id] || null;
    teamsOut[id] = { id, abbr: t.abbreviation, name: t.displayName, short: t.shortDisplayName || t.location, mascot: t.name || '', color: '#' + (t.color || '444444'), alt: '#' + (t.alternateColor || '888888'),
      fbs: FBS[CUR].has(id), conf, ap: (ranks[id] || {}).ap || null, cfp: (ranks[id] || {}).cfp || null, logo };
  });

  // full live rosters (limit=300 — the default response silently stops at 100 players)
  const liveRoster = {};
  await pool([...teamIds], 8, async id => { const j = await fetchJson(`${ESPN}/teams/${id}/roster?limit=300`); liveRoster[id] = j ? (j.athletes || []).flatMap(gr => (gr.items || []).map(a => ({ a, grp: gr.position }))) : null; });
  const rosterFail = Object.entries(liveRoster).filter(([, v]) => !v).map(([k]) => k);
  if (rosterFail.length) log(`  ! roster fetch failed for ${rosterFail.length} teams: ${rosterFail.join(',')}`);

  // venues: coordinates for the weather lookup (venues.json, geocoded once)
  const venues = loadVenues();
  for (const g of schedule) await ensureVenue(venues, g.venue);
  saveVenues(venues);
  const missingVenue = schedule.filter(g => g.venue.id && (!venues[g.venue.id] || venues[g.venue.id].lat == null)).map(g => g.venue.name);
  log(`  venues: ${Object.keys(venues).length} cached (${geocodedThisRun} geocoded this run)${missingVenue.length ? `; no coordinates for ${missingVenue.join(', ')}` : ''}`);

  // ==========================================================================
  // Player rows: join live rosters to play-by-play by athlete ID. Every field the app's distribute() reads is built
  // with the same definitions the backtest uses (usage recency, touches per game, presumed starting QB).
  // ==========================================================================
  const Clive = { ...C, NEWPRIOR: newPriorFor(CUR - 1, C, MODEL.recVol) };
  log(`  new-player prior (${CUR - 1}, per team-game): ${Object.entries(Clive.NEWPRIOR).map(([k, v]) => `${k} rush ${v.rush} rec ${v.rec}`).join('; ')}`);
  const SKILL = new Set(['QB', 'RB', 'FB', 'WR', 'TE', 'ATH']);
  const curGamesOf = team => (byTeam.get(team) || []).filter(tg => tg.s === CUR);
  const rostersOut = {};
  const onRoster = new Map();   // athleteId -> team
  for (const [team, items] of Object.entries(liveRoster)) for (const { a } of items || []) onRoster.set(a.id, team);
  const priorSeasons = gotSeasons.filter(s => s !== CUR);
  const touchesOf = r => r.rushAtt - r.kneel + r.tgt;
  let skillN = 0, skillHist = 0, xfers = 0;
  for (const team of teamIds) {
    const items = liveRoster[team]; if (!items) continue;
    const tgs = curGamesOf(team);
    const lastTG = tgs[tgs.length - 1] || null;
    const lastStarters = lastTG ? starters.get(lastTG.gid) || null : null;
    let teamTouches = 0; for (const tg of tgs) teamTouches += tg.carries + tg.tgt;
    const rows = [];
    for (const { a } of items) {
      const posRaw = (a.position && a.position.abbreviation) || '';
      const recs = playerGames.get(a.id) || [];
      const ag = aggregate(recs, liveW);
      const hasUsage = ag.rushAtt + ag.tgt + ag.drop > 0, isRet = ag.kr + ag.pr > 0;
      if (!SKILL.has(posRaw) && !hasUsage && !isRet) continue;
      // bucket: listed position; ATH / defenders with offensive usage go by what they actually do
      const bk = posBucket(posRaw) || usageBucket(ag);
      const retOnly = !bk;
      const cur = recs.filter(r => r.s === CUR && r.team === team);
      const prior = recs.filter(r => priorSeasons.includes(r.s));
      const hist = prior.some(r => FBS[r.s].has(r.team)) ? 'fbs' : prior.length ? 'fcs' : 'none';
      const lastPrior = prior.length ? prior.reduce((x, y) => (y.s > x.s || (y.s === x.s && y.wk > x.wk) ? y : x)) : null;
      const xferFrom = lastPrior && lastPrior.team !== team ? lastPrior.team : null;
      // usage recency (the availability signal): team games since he last touched the ball / returned a kick
      let lastTouchAgo = null;
      for (let i = tgs.length - 1; i >= 0; i--) { const pg = pgIndex.get(a.id + '|' + tgs[i].gid); if (pg && pg.team === team && (touchedRec(pg) || pg.kr > 0 || pg.pr > 0)) { lastTouchAgo = tgs.length - 1 - i; break; } }
      const lastPG = lastTG ? pgIndex.get(a.id + '|' + lastTG.gid) : null;
      const curTouches = cur.reduce((s, r) => s + touchesOf(r), 0);
      const starts = tgs.filter(tg => starters.has(tg.gid) && starters.get(tg.gid).has(a.id)).length;
      const sc = bk ? CFBModel.channels(ag, bk, MODEL, Clive) : { rush: MODEL.eps, rec: MODEL.eps };
      const d = ag.g + MODEL.shrinkGames;
      if (SKILL.has(posRaw)) { skillN++; if (recs.length) skillHist++; }
      if (xferFrom && cur.length) xfers++;
      rows.push({
        id: a.id, name: a.fullName || a.displayName, pos: bk || posRaw, listed: posRaw, jersey: a.jersey || '', cls: (a.experience && a.experience.abbreviation) || '',
        status: 'ACT', retOnly,
        rushScore: r4(sc.rush), recScore: r4(sc.rec), ret: +(ag.kr + ag.pr).toFixed(2),
        touchPg: +(ag.g ? touchesOf(ag) / ag.g : 0).toFixed(2), hasHist: ag.g > 0, lastTouchAgo,
        games: +ag.g.toFixed(1), gCur: cur.filter(r => touchedRec(r)).length,
        rushPg: +(ag.rushAtt / d).toFixed(2), tgtPg: +(ag.tgt / d).toFixed(2), glPg: r3(ag.glCarry / d), rzTgtPg: r3(ag.rzTgt / d),
        rushTDpg: r3(ag.rushTD / d), recTDpg: r3(ag.recTD / d), stTD: +ag.stTD.toFixed(2),
        touchShare: teamTouches ? r3(curTouches / teamTouches) : null,
        lastTouches: lastPG ? touchesOf(lastPG) : 0, dropCur: cur.reduce((s, r) => s + r.drop, 0), dropAll: +ag.drop.toFixed(1),
        starts, startedLast: lastStarters ? lastStarters.has(a.id) : null,
        hist, xferFrom, thin: ag.g < 3,
      });
    }
    // presumed starting QB — the same rule the backtest uses (pickStarterQB / gameQBof)
    const qbs = rows.filter(r => r.pos === 'QB'), qbIds = new Set(qbs.map(q => q.id));
    const gq = tgs.map(tg => gameQBof(tg.gid, team, pid => qbIds.has(pid))), lastQB = gq.length ? gq[gq.length - 1] : null;
    for (const q of qbs) q.ledGames = gq.filter(id => id === q.id).length;
    const starterId = pickStarterQB(qbs.map(q => ({ id: q.id, isLast: q.id === lastQB, led: q.ledGames, dropCur: q.dropCur, dropAll: q.dropAll })));
    for (const q of qbs) { q.starterQB = q.id === starterId; if (!q.starterQB) q.backupQB = true; }
    rows.sort((x, y) => (y.rushScore + y.recScore) - (x.rushScore + x.recScore));
    // roster confidence: how much we can trust this week's usage picture
    const qbBattle = new Set(gq.filter(Boolean)).size > 1;
    const reasons = [];
    if (!teamsOut[team].fbs) reasons.push('FCS team (history only from games vs FBS)');
    if (!tgs.length) reasons.push('no games yet this season');
    else if (tgs.length < 2) reasons.push('only 1 game this season');
    if (qbBattle) reasons.push('different QBs have led games this season');
    const thinShare = teamTouches ? rows.filter(r => r.hist === 'none').reduce((s, r) => s + (r.touchShare || 0), 0) : 0;
    if (thinShare >= 0.35) reasons.push(`${Math.round(thinShare * 100)}% of touches to players with no prior history`);
    teamsOut[team].rosterConf = reasons.length ? 'low' : 'ok';
    teamsOut[team].confNotes = reasons;
    rostersOut[team] = rows;
    if (!profiles[team]) profiles[team] = { offTDpg: r3(leagueOffTDpg), rushShare: 0.5, playsPg: 70, giveawayPg: DST0.leagueGive, gamesCur: 0, record: '0-0', ppgCur: null, oppPpgCur: null, def: { tdAllowPg: r3(leagueOffTDpg), rushAllowShare: 0.5, byPos: { ...leagueByPos } }, noHistory: true };
  }

  // ---- join report: of the players who touched the ball for an FBS team this season, who is on that live roster? ----
  const producers = new Map();   // pid|team -> touches
  for (const [pid, recs] of playerGames) for (const r of recs) if (r.s === CUR && FBS[CUR].has(r.team)) { const k = pid + '|' + r.team; producers.set(k, (producers.get(k) || 0) + r.rushAtt - r.kneel + r.tgt + r.drop); }
  let pN = 0, pOn = 0, pOther = 0, tAll = 0, tOn = 0; const unmatched = [];
  for (const [k, n] of producers) { const [pid, team] = k.split('|'); if (!n) continue; pN++; tAll += n; const t = onRoster.get(pid);
    if (t === team) { pOn++; tOn += n; } else { if (t) pOther++; unmatched.push({ id: pid, name: pbpName.get(pid) || athName.get(pid) || pid, team, touches: n, now: t || null }); } }
  unmatched.sort((a, b) => b.touches - a.touches);
  const join = { producers: pN, onTeamRoster: pOn, onOtherRoster: pOther, rate: r3(pOn / pN), touchRate: r3(tOn / tAll), skillRosterPlayers: skillN, withHistory: skillHist, transfersProducing: xfers, unmatched: unmatched.slice(0, 40) };
  log(`  ID join: ${pOn}/${pN} ${CUR} producers are on their team's live ESPN roster (${pct(pOn, pN)}; ${pct(tOn, tAll)} of touches); ${pOther} on another roster; unmatched: ${unmatched.length}`);
  if (unmatched.length) log(`    unmatched (top): ${unmatched.slice(0, 12).map(u => `${u.name} [${teamsOut[u.team] ? teamsOut[u.team].abbr : u.team}] ${u.touches}`).join('; ')}`);
  log(`  skill-position roster players: ${skillN}, with any play-by-play history ${skillHist} (${pct(skillHist, skillN)}); transfers already producing for the new school: ${xfers}`);
  const lowConf = Object.values(teamsOut).filter(t => t.rosterConf === 'low');
  log(`  low-confidence rosters: ${lowConf.length} (${lowConf.slice(0, 8).map(t => `${t.abbr}: ${t.confNotes.join(', ')}`).join(' | ')}${lowConf.length > 8 ? ' ...' : ''})`);

  // ==========================================================================
  // Snapshot
  // ==========================================================================
  const teamList = Object.keys(teamsOut).filter(id => rostersOut[id]).sort((a, b) => teamsOut[a].name.localeCompare(teamsOut[b].name));
  const confs = {}; for (const id of teamList) { const c = teamsOut[id].conf; if (c && CONF_NAME[c]) confs[c] = CONF_NAME[c]; }
  const venuesOut = {}; for (const g of schedule) if (g.venue.id && venues[g.venue.id]) venuesOut[g.venue.id] = venues[g.venue.id];
  const curWeek = Math.max(0, ...[...games.values()].filter(g => g.s === CUR).map(g => g.wk));
  const snapshot = {
    meta: {
      builtAt: new Date().toISOString(), seasons: gotSeasons, curSeason: CUR, lastWeekInData: curWeek, slateWeek: weekInfo ? weekInfo.label : null,
      seasonWeights: SEASON_WEIGHT, nsims: SIM_META.nsims,
      source: 'sportsdataverse espn_cfb_pbp (ESPN play-by-play) + ESPN rosters/lines/rankings/logos + Open-Meteo',
      join, tdCheck,
    },
    constants: { ...Clive, KAPPA, leagueOffTDpg: r3(leagueOffTDpg), leagueByPos, MODEL, DST_MODEL, ROLE_CAL: btOut.roleCal },
    teams: teamsOut, teamList, confs,
    otherTeams: (() => { const o = {}; for (const rows of Object.values(rostersOut)) for (const r of rows) if (r.xferFrom && !teamsOut[r.xferFrom]) o[r.xferFrom] = allAbbr[r.xferFrom] || null; return o; })(),
    profiles: Object.fromEntries(teamList.map(id => [id, profiles[id]])),
    rosters: Object.fromEntries(teamList.map(id => [id, rostersOut[id]])),
    schedule, venues: venuesOut,
    backtest: btOut.backtest, dstBacktest: btOut.dstBacktest,
  };
  const jsonPath = path.join(__dirname, 'cfb-td-snapshot.json');
  fs.writeFileSync(jsonPath, JSON.stringify(snapshot));
  log(`  wrote ${jsonPath} (${(fs.statSync(jsonPath).size / 1e6).toFixed(2)} MB)`);
  const tplPath = path.join(__dirname, 'cfb-td-predictor.template.html');
  if (fs.existsSync(tplPath)) {
    const out = fs.readFileSync(tplPath, 'utf8')
      .replace('/*__MODEL__*/', () => MODEL_SRC)
      .replace('/*__SNAPSHOT__*/', () => 'window.__SNAPSHOT__ = ' + JSON.stringify(snapshot) + ';');
    const outPath = path.join(__dirname, 'cfb-td-predictor.html');
    fs.writeFileSync(outPath, out);
    log(`  wrote ${outPath} (${(fs.statSync(outPath).size / 1e6).toFixed(2)} MB)`);
  } else log('  (no template yet — wrote JSON only)');
  log(`  teams shipped: ${teamList.length} (${teamList.filter(id => teamsOut[id].fbs).length} FBS), player rows: ${Object.values(rostersOut).reduce((a, r) => a + r.length, 0)}`);
  log(`=== done in ${((Date.now() - t0) / 1000).toFixed(0)} s ===\n`);
})().catch(e => { console.error('BUILD FAILED:', e); process.exit(1); });
