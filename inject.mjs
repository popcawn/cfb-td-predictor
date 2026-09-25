// Template-only dev loop: re-inject the existing snapshot + model into the template (no data rebuild), then parse-check.
import fs from 'node:fs';
const tpl = fs.readFileSync('cfb-td-predictor.template.html', 'utf8');
const out = tpl.replace('/*__MODEL__*/', () => fs.readFileSync('cfb-model.js', 'utf8'))
  .replace('/*__SNAPSHOT__*/', () => 'window.__SNAPSHOT__ = ' + fs.readFileSync('cfb-td-snapshot.json', 'utf8') + ';');
for (const [i, m] of [...out.matchAll(/<script>([\s\S]*?)<\/script>/g)].entries()) new Function(m[1]);
fs.writeFileSync('cfb-td-predictor.html', out);
console.log(`cfb-td-predictor.html ${(out.length / 1e6).toFixed(2)} MB — scripts parse OK`);
