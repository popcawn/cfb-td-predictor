/*
 * cfb-model.js — the model's probability math, shared VERBATIM by the build (live player scores + backtest)
 * and the app (the build injects this file's source into the HTML). Anything that changes a probability
 * belongs here and is switched by MODEL, so the backtest always scores exactly what ships.
 * Plain ES5-ish script (no imports/exports): it defines one global, CFBModel.
 */
var CFBModel = (function () {
  // ---- count distributions -------------------------------------------------------------------------------
  // Team offensive TDs are gamma-Poisson (negative binomial, shape `size`). Thinning that by a player's fixed share
  // keeps the shape, so a player's offensive-TD count is NB(size) with his expected TDs as the mean.
  function nbP0(exp, size) { return Math.pow(1 + exp / size, -size); }
  // P(scores at least once): offensive TDs (NB) and kick/punt-return TDs (Poisson, expST) are independent
  function anytime(expOff, expST, size) { return 1 - nbP0(expOff, size) * Math.exp(-(expST || 0)); }
  // P(2+ TDs) from offensive TDs (return TDs are too rare to matter here)
  function twoPlus(exp, size) {
    var q = size / (size + exp), p0 = Math.pow(q, size), p1 = size * (1 - q) * p0;
    return Math.max(0, 1 - p0 - p1);
  }

  // ---- team level -------------------------------------------------------------------------------------------
  // Offensive TDs per implied point (kappa), optionally leaning with the side's margin (points favored by).
  function kappaAt(K, margin, M) { return K * Math.max(0.85, Math.min(1.15, 1 + (M.kSlope || 0) * margin / 10)); }
  // Expected offensive TDs for a side: Vegas implied points (total/2 + margin/2) x kappa.
  function expOffTD(total, margin, K, M) { return Math.max(0.4, (total / 2 + margin / 2) * kappaAt(K, margin, M)); }
  // Share of a team's offensive TDs on the ground: its recency-weighted TD split, leaned toward the run in
  // wind / rain / snow (outdoor games only).
  function rushShare(base, M, wx) {
    var rs = base;
    if (M.weather && wx && wx.outdoor) {
      if (wx.wind > 12) rs += Math.min(0.09, (wx.wind - 12) * 0.005);
      rs += wx.precip === 'snow' ? 0.05 : wx.precip === 'rain' ? 0.03 : 0;
    }
    return Math.max(0.24, Math.min(0.80, rs));
  }
  // Defensive-TD rate (pick-6 / fumble return) for a defense: base x e^(slope x points favored) x
  // (opponent giveaways per game / league)^giveExp. Kick/punt returns are NOT here — they pay the returner's prop.
  function defTDrate(margin, oppGive, D) {
    return D.base * Math.exp(D.slope * margin) * Math.pow((oppGive || D.leagueGive) / D.leagueGive, D.giveExp);
  }

  // ---- player level -----------------------------------------------------------------------------------------
  // Per-game scoring weights from recency-weighted usage counts a = {g, rushAtt, kneel, rushTD, glCarry, tgt,
  // rec, rzTgt, recTD, airY, airTgt}. Players with no history at all get the new-player prior for their position.
  function channels(a, bk, M, C) {
    var pr = M.prior[bk] || M.prior.X;
    if (!a || !(a.g > 0)) {
      var np = M.newPrior === 'emp' && C.NEWPRIOR && C.NEWPRIOR[bk] ? C.NEWPRIOR[bk] : pr;
      // the measured new-player prior already includes the zeros, so the universal floor (eps) is optional here
      var e = M.newEps === false ? 0.0005 : M.eps;
      return { rush: np.rush + e, rec: np.rec + e };
    }
    var d = a.g + M.shrinkGames;
    var sp = M.shrinkPrior ? M.ps * M.shrinkGames / d * (M.priorScale || 1) : 0;
    var carries = a.rushAtt - (M.noKneel ? a.kneel : 0);
    var rush = 0.40 * a.rushTD / d + 0.40 * (a.glCarry / d) * C.GLCONV + pr.rush * sp;
    var carry = 0.20 * (carries / d) * C.RUSHTDATT * (bk === 'QB' ? M.qbCarry : 1);
    // pass-volume term: targets, or catches (2023-24 incompletions rarely name the receiver)
    var vol = M.recVol === 'rec' ? (a.rec / d) * C.RECTD : (a.tgt / d) * C.TGTTD;
    // air yards per target that HAS air yards (so seasons without the field don't read as zero depth)
    var air = M.air && a.airTgt > 0 ? (a.airY / a.airTgt) * (a.tgt / d) * C.AIRYDTD : 0;
    var rec = 0.35 * a.recTD / d + 0.35 * (a.rzTgt / d) * C.RZTGTCONV + (M.air ? 0.20 : 0.30) * vol + (M.air ? 0.10 * air : 0) + pr.rec * sp;
    // college QBs run far more (and score more on the ground) than their usage profile alone says
    var qr = bk === 'QB' ? (M.qbRush || 1) : 1;
    return { rush: (rush + carry) * qr + M.eps, rec: rec + M.eps };
  }
  // Availability from usage recency (no college injury feed exists): gamesAgo = team games since the player's last
  // touch (0 = touched last game), null = no touch yet this season; teamGames = team games played this season.
  function availFactor(gamesAgo, teamGames, M) {
    if (!M.avail) return 1;
    if (gamesAgo == null) return teamGames >= M.availGrace ? M.aNone : 1;
    return gamesAgo === 0 ? 1 : gamesAgo === 1 ? M.a1 : M.a2;
  }
  // Usage tier (the college stand-in for snap share): touches (carries + targets) per game played, recency-weighted.
  function usageTier(bk, touchPg, hasHist) {
    if (bk === 'QB') return 'qb';
    if (!hasHist) return 'new';
    return touchPg >= 10 ? 'feature' : touchPg >= 5 ? 'regular' : touchPg >= 2 ? 'rotation' : 'fringe';
  }

  // Distribute one side's expected TDs over its players.
  //   cands: [{rush, rec, ret, bk, qbStarter (true/false/null), gamesAgo, touchPg, hasHist}]
  //   g:     {total, margin, rushBase, wx, teamGames, K, stBase}
  // Returns [{expOff, expST, p (anytime), p2 (2+ TDs), tier, share}] in the same order.
  function distribute(cands, g, M) {
    var n = cands.length, i, wr = new Array(n), wc = new Array(n), sR = 0, sC = 0, sT = 0;
    for (i = 0; i < n; i++) {
      var c = cands[i], f = availFactor(c.gamesAgo, g.teamGames, M);
      if (c.bk === 'QB' && c.qbStarter === false) f *= M.qbBackup;      // one starting QB; the rest are backups
      wr[i] = c.rush * f; wc[i] = c.rec * f; sR += wr[i]; sC += wc[i]; sT += c.ret || 0;
    }
    // Garbage time: the bigger the spread, the more a team's TDs spread to the depth chart. Flatten the shares
    // (power < 1) as |margin| grows past gtFrom points.
    var am = Math.abs(g.margin), pw = 1;
    if (M.gt && am > M.gtFrom && (M.gtSide === 'both' || g.margin > 0)) pw = 1 / (1 + M.gt * (am - M.gtFrom) / 14);
    if (pw !== 1) {
      var tR = 0, tC = 0;
      for (i = 0; i < n; i++) { wr[i] = sR ? Math.pow(wr[i] / sR, pw) : 0; wc[i] = sC ? Math.pow(wc[i] / sC, pw) : 0; tR += wr[i]; tC += wc[i]; }
      sR = tR; sC = tC;
    }
    var expOff = expOffTD(g.total, g.margin, g.K, M), rs = rushShare(g.rushBase, M, g.wx), out = new Array(n);
    for (i = 0; i < n; i++) {
      var sh = rs * (sR ? wr[i] / sR : 0) + (1 - rs) * (sC ? wc[i] / sC : 0);
      var eo = expOff * sh, es = sT ? g.stBase * (cands[i].ret || 0) / sT : 0;
      out[i] = { expOff: eo, expST: es, share: sh, p: anytime(eo, es, M.nbSize), p2: Math.min(twoPlus(eo, M.nbSize) * (M.twoCal || 1), anytime(eo, es, M.nbSize)),
        tier: usageTier(cands[i].bk, cands[i].touchPg, cands[i].hasHist) };
    }
    return out;
  }
  // Post-hoc role calibration (cross-fitted in the backtest): scales a tier's probabilities by actual/predicted.
  function roleCal(p, tier, RC) { var f = RC && RC[tier] ? RC[tier] : 1; return Math.min(0.97, p * f); }

  return { nbP0: nbP0, anytime: anytime, twoPlus: twoPlus, kappaAt: kappaAt, expOffTD: expOffTD, rushShare: rushShare,
    defTDrate: defTDrate, channels: channels, availFactor: availFactor, usageTier: usageTier, distribute: distribute, roleCal: roleCal };
})();
