#!/usr/bin/env node
/*
 * build-market-lines.mjs
 * -------------------------------------------------------------------------
 * Scrapes ESPN's free historical "Anytime Touchdown Scorer" boards (which players the sportsbook actually made an
 * anytime-TD market on) for every FBS game of a season, and writes market_lines_<season>.json keyed by ESPN game id.
 * build-cfb-td-snapshot.mjs picks it up and reports the model's calibration restricted to the book's player universe.
 *
 * ESPN exposes the LINE (target 0.5 = 1+ TD), NOT the price — so this is a "who did the book price, and is the model
 * calibrated on them" check, not a beat-the-odds check. ESPN BET boards are archived for part of 2025 only.
 * Game and athlete ids are ESPN's, the same ids the play-by-play uses, so nothing needs mapping.
 *
 * Usage:  node build-market-lines.mjs [season]        (default 2025)   FORCE=1 to ignore the existing file
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SEASON = +(process.argv[2]) || 2025;
const PROVIDER = 58;   // ESPN BET
const OUT = path.join(__dirname, `market_lines_${SEASON}.json`);
const ESPN = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football';
const CORE = 'https://sports.core.api.espn.com/v2/sports/football/leagues/college-football';

async function fetchJson(url, tries = 3) {
  for (let a = 0; a < tries; a++) {
    try { const r = await fetch(url); if (r.ok) return await r.json(); if (r.status === 404 || r.status === 400) return null; } catch { }
    await new Promise(res => setTimeout(res, 600 * (a + 1)));
  }
  return null;
}
async function pool(items, n, fn) {
  let i = 0; await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (i < items.length) { const k = i++; await fn(items[k], k); } }));
}
async function board(id) {
  const ids = new Set(); let page = 1, pages = 1;
  do {
    const j = await fetchJson(`${CORE}/events/${id}/competitions/${id}/odds/${PROVIDER}/propBets?limit=1000&page=${page}`);
    if (!j) break;
    pages = j.pageCount || 1;
    for (const it of j.items || []) if (it.type && it.type.id === '31' && it.current && it.current.target && it.current.target.value <= 0.5) {
      const m = it.athlete && it.athlete.$ref && it.athlete.$ref.match(/athletes\/(\d+)/); if (m) ids.add(m[1]);
    }
    page++;
  } while (page <= pages);
  return [...ids];
}

(async function main() {
  if (fs.existsSync(OUT) && !process.env.FORCE) { console.log(`${OUT} exists (FORCE=1 to rebuild)`); return; }
  console.log(`\n=== ESPN BET anytime-TD boards, college ${SEASON} ===`);
  const sb = await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&dates=${SEASON}`);
  const events = [];
  for (const c of (sb && sb.leagues && sb.leagues[0].calendar) || []) {
    if (c.value !== '2' && c.value !== '3') continue;
    for (const e of c.entries || []) {
      const j = await fetchJson(`${ESPN}/scoreboard?groups=80&limit=300&dates=${SEASON}&week=${e.value}&seasontype=${c.value}`);
      for (const ev of (j && j.events) || []) events.push({ id: ev.id, week: c.value === '3' ? 'post' : +e.value, name: ev.shortName });
    }
  }
  console.log(`${events.length} games; scraping boards ...`);
  const out = {}; let done = 0, withBoard = 0;
  await pool(events, 6, async ev => {
    const ids = await board(ev.id);
    out[ev.id] = { week: ev.week, name: ev.name, listed: ids.length, market: ids };
    if (ids.length) withBoard++;
    if (++done % 100 === 0) console.log(`  ${done}/${events.length} (${withBoard} with an anytime-TD board)`);
  });
  const byWeek = {}; for (const g of Object.values(out)) { const w = byWeek[g.week] || (byWeek[g.week] = [0, 0]); w[1]++; if (g.listed) w[0]++; }
  console.log(`  boards by week (with/total): ${Object.entries(byWeek).map(([w, [a, b]]) => `${w}:${a}/${b}`).join(' ')}`);
  const meta = { season: SEASON, provider: 'ESPN BET (id 58)', scrapedAt: new Date().toISOString(), games: events.length, withBoard };
  fs.writeFileSync(OUT, JSON.stringify({ meta, games: out }));
  console.log(`wrote ${OUT} (${withBoard} of ${events.length} games have a board)\n=== done ===`);
})().catch(e => { console.error('SCRAPE FAILED:', e); process.exit(1); });
