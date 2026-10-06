/* Project #2 tabs: Zipf analysis, word2vec, edit distance / spelling correction */
(() => {
  const { $, $$, esc, fmt, api, state, showTab, doSearch } = window.IR;
  const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)'];
  const PREV = { B: 'A', C: 'B', D: 'C', E: 'B' };   // which condition each one is built on (E branches off B, skipping the stop-word step)
  const f2 = (v, d = 3) => Number(v).toFixed(d);
  const pct = v => (v * 100).toFixed(1) + '%';
  const SVGNS = 'http://www.w3.org/2000/svg';

  /* ------------------------------------------------------------------ *
   * x/y plot (inline SVG): lines or dots, linear or log axes, hover tooltip
   * cfg = { series: [{ name, color, pts: [[x, y, label?]], mode: 'line'|'dots'|'fit', r }],
   *         xlog, ylog, xlabel, ylabel, height, vlines: [{ x, label }], hover: 'x'|'nearest',
   *         labels (draw point labels), tipX(x) -> text, tipY(y) -> text,
 *         y2: { log, zero, label } scale of the series with axis: 'r' (drawn against a right-hand axis),
 *         strips: [{ from, to, color, label, mark }] x-ranges drawn as bars above the plot }
   * ------------------------------------------------------------------ */
  function niceTicks(a, b, n = 5) {
    const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw)), norm = raw / mag;
    const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  const short = v => { const a = Math.abs(v); return a >= 1e6 ? v / 1e6 + 'M' : a >= 1e3 ? v / 1e3 + 'k' : String(+v.toPrecision(3)); };

  function plot(el, cfg) {
    const y2 = cfg.y2 || {}, strips = cfg.strips || [];
    const W = cfg.width || 640, H = cfg.height || 340, P = { l: 58, r: cfg.y2 ? 58 : 18, t: 16 + strips.length * 10, b: 44 };
    const tx = v => (cfg.xlog ? Math.log10(v) : v), ty = v => (cfg.ylog ? Math.log10(v) : v), ty2 = v => (y2.log ? Math.log10(v) : v);
    const all = cfg.series.flatMap(s => s.pts);
    if (!all.length) { el.innerHTML = '<p class="muted">no data</p>'; return; }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, r0 = Infinity, r1 = -Infinity;
    cfg.series.forEach(se => se.pts.forEach(p => {
      const x = tx(p[0]); if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (se.axis === 'r') { const y = ty2(p[1]); if (y < r0) r0 = y; if (y > r1) r1 = y; } else { const y = ty(p[1]); if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }));
    if (!cfg.xlog && cfg.xzero) x0 = Math.min(0, x0);
    if (!cfg.ylog && cfg.yzero) y0 = Math.min(0, y0);
    const px = (x1 - x0 || 1) * 0.03, py = (y1 - y0 || 1) * 0.05;
    if (cfg.xlog || !cfg.xzero) x0 -= px; x1 += px;
    if (cfg.ylog || !cfg.yzero) y0 -= py; y1 += py;
    const sx = v => P.l + (tx(v) - x0) / (x1 - x0) * (W - P.l - P.r);
    const sy = v => H - P.b - (ty(v) - y0) / (y1 - y0) * (H - P.t - P.b);
    if (y2.zero && !y2.log) r0 = Math.min(0, r0);
    r1 += (r1 - r0 || 1) * 0.05;
    const sy2 = v => H - P.b - (ty2(v) - r0) / (r1 - r0) * (H - P.t - P.b), syOf = se => (se.axis === 'r' ? sy2 : sy);
    const ticks = (log, a, b) => {
      if (!log) return niceTicks(a, b).map(v => [v, short(v)]);
      const out = [];
      for (let e = Math.ceil(a - 1e-9); e <= Math.floor(b + 1e-9); e++) out.push([10 ** e, e < 0 ? String(+(10 ** e).toPrecision(1)) : short(10 ** e)]);
      return out;
    };
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block" role="img" aria-label="${esc(cfg.title || '')}">`;
    ticks(cfg.ylog, y0, y1).forEach(([v, l]) => {
      s += `<line x1="${P.l}" x2="${W - P.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--line)"/><text x="${P.l - 8}" y="${sy(v) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">${l}</text>`;
    });
    ticks(cfg.xlog, x0, x1).forEach(([v, l]) => {
      s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${P.t}" y2="${H - P.b}" stroke="var(--line)"/><text x="${sx(v)}" y="${H - P.b + 16}" font-size="11" text-anchor="middle" fill="var(--muted)">${l}</text>`;
    });
    if (cfg.y2 && r1 > r0) {
      ticks(y2.log, r0, r1).forEach(([, l], i, a) => { const v = y2.log ? Math.log10(a[i][0]) : a[i][0], y = H - P.b - (v - r0) / (r1 - r0) * (H - P.t - P.b);
        s += `<text x="${W - P.r + 8}" y="${y + 4}" font-size="11" fill="var(--muted)">${l}</text>`; });
      s += `<text transform="translate(${W - 8} ${(P.t + H - P.b) / 2}) rotate(90)" font-size="12" text-anchor="middle" fill="var(--muted)">${esc(y2.label || '')}</text>`;
    }
    strips.forEach((b, i) => {
      const y = 8 + i * 10, xa = sx(b.from), xb = Math.max(sx(b.to), xa + 2);
      s += `<rect x="${xa}" y="${y}" width="${xb - xa}" height="6" rx="2" fill="${b.color}"/><text x="${xa - 5}" y="${y + 6}" font-size="10" font-weight="700" text-anchor="end" fill="${b.color}">${esc(b.label || '')}</text>`;
      if (b.mark) s += `<circle cx="${sx(b.mark)}" cy="${y + 3}" r="3.5" fill="var(--card)" stroke="${b.color}" stroke-width="2"/>`;
    });
    s += `<text x="${(P.l + W - P.r) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="var(--muted)">${esc(cfg.xlabel || '')}</text>`;
    s += `<text transform="translate(14 ${(P.t + H - P.b) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="var(--muted)">${esc(cfg.ylabel || '')}</text>`;
    (cfg.shade || []).forEach(b => {
      s += `<rect x="${sx(b.from)}" y="${P.t}" width="${Math.max(0, sx(b.to) - sx(b.from))}" height="${H - P.t - P.b}" fill="${b.color}" fill-opacity="0.13"/>`;
    });
    (cfg.vlines || []).forEach(v => {
      s += `<line x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${P.t}" y2="${H - P.b}" stroke="var(--muted)" stroke-dasharray="3 4"/>`;
    });
    (cfg.bands || []).forEach(b => {
      s += `<text x="${(sx(b.from) + sx(b.to)) / 2}" y="${P.t + 12}" font-size="11" text-anchor="middle" fill="var(--muted)">${esc(b.label)}</text>`;
    });
    cfg.series.forEach(se => {
      const sy = syOf(se);
      if (se.mode === 'dots') {
        s += se.pts.map(p => `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${se.r || 3}" fill="${se.color}" fill-opacity="${se.opacity || 0.75}"/>`).join('');
      } else {
        const d = se.pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join('');
        s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || (se.mode === 'fit' ? 1.5 : 2)}" stroke-opacity="${se.opacity || 1}" ${se.mode === 'fit' ? 'stroke-dasharray="6 4"' : ''} stroke-linejoin="round"/>`;
      }
    });
    if (cfg.labels) {                                    // point labels, skipping the ones that would collide
      const boxes = [];
      cfg.series.flatMap(se => se.pts.map(p => [p, se])).sort(([p], [q]) => (q[3] ? 1 : 0) - (p[3] ? 1 : 0)).forEach(([p, se]) => {      // emphasised labels first
        if (!p[2]) return;
        const x = sx(p[0]) + 6, y = syOf(se)(p[1]) + 4, w = String(p[2]).length * 6.2, box = [x, y - 10, x + w, y + 2];
        if (x + w > W - 2 || boxes.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
        boxes.push(box);
        s += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="11" fill="var(--ink)" ${p[3] ? 'font-weight="700"' : ''}>${esc(p[2])}</text>`;
      });
    }
    s += `<line class="xh" y1="${P.t}" y2="${H - P.b}" stroke="var(--muted)" visibility="hidden"/><circle class="ring" r="6" fill="none" stroke="var(--ink)" stroke-width="2" visibility="hidden"/>`;
    s += `<rect class="hit" x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}" fill="transparent"/></svg>`;
    const named = cfg.series.filter(se => se.name && se.legend !== false);
    if (named.length >= 2) {
      s += '<div class="legend plot-legend">' + named.map(se => `<span><i class="${se.mode === 'dots' ? 'dot' : 'ln'}${se.mode === 'fit' ? ' dash' : ''}" style="--c:${se.color}"></i>${esc(se.name)}</span>`).join('') + '</div>';
    }
    el.classList.add('plot');
    el.innerHTML = s + '<div class="tip hidden"></div>';

    const svg = $('svg', el), tip = $('.tip', el), xh = $('.xh', el), ring = $('.ring', el);
    const tipX = cfg.tipX || (v => `${cfg.xlabel || 'x'} ${fmt(v)}`), tipY = cfg.tipY || (v => fmt(v));
    const hide = () => { tip.classList.add('hidden'); xh.setAttribute('visibility', 'hidden'); ring.setAttribute('visibility', 'hidden'); };
    $('.hit', el).addEventListener('pointerleave', hide);
    $('.hit', el).addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), k = W / r.width, mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
      tip.textContent = '';
      const row = (color, value, label) => {
        const d = document.createElement('div');
        if (color) { const key = document.createElement('i'); key.style.background = color; d.appendChild(key); }
        const b = document.createElement('b'); b.textContent = value; d.appendChild(b);
        if (label) { const sp = document.createElement('span'); sp.textContent = ' ' + label; d.appendChild(sp); }
        tip.appendChild(d);
      };
      if (cfg.hover === 'nearest') {
        let best = null, bd = Infinity, bs = null;
        cfg.series.forEach(se => se.pts.forEach(p => { const d = (sx(p[0]) - mx) ** 2 + (syOf(se)(p[1]) - my) ** 2; if (d < bd) { bd = d; best = p; bs = se; } }));
        if (!best) return hide();
        ring.setAttribute('cx', sx(best[0])); ring.setAttribute('cy', syOf(bs)(best[1])); ring.setAttribute('visibility', 'visible');
        (cfg.tipRows ? cfg.tipRows(best, bs) : [[bs.color, String(best[2] || ''), ''], [null, tipY(best[1]), tipX(best[0])]]).forEach(a => row(...a));
      } else {
        let anchor = null;
        cfg.series.forEach(se => {
          if (se.tip === false) return;
          let best = se.pts[0], bd = Infinity;
          se.pts.forEach(p => { const d = Math.abs(sx(p[0]) - mx); if (d < bd) { bd = d; best = p; } });
          if (!anchor) { anchor = best; const h = document.createElement('div'); h.className = 'tip-h'; h.textContent = tipX(best[0]) + (best[2] ? ' · ' + best[2] : ''); tip.appendChild(h); }
          row(se.color, (se.tipY || tipY)(best[1]), se.name || '');
        });
        if (!anchor) return hide();
        xh.setAttribute('x1', sx(anchor[0])); xh.setAttribute('x2', sx(anchor[0])); xh.setAttribute('visibility', 'visible');
      }
      tip.classList.remove('hidden');
      const er = el.getBoundingClientRect(), left = ev.clientX - er.left, flip = left > er.width * 0.6;
      tip.style.left = (flip ? left - tip.offsetWidth - 14 : left + 14) + 'px';
      tip.style.top = Math.max(0, ev.clientY - er.top - tip.offsetHeight - 8) + 'px';
    });
  }

  /* ================================================================== *
   * Zipf tab
   * ================================================================== */
  const zs = { data: null, cond: 'B', scale: 'logy', terms: '', table: null, res: null, resAll: null, upper: null, lower: null, nonum: false, map: null, mapThr: {} };
  const SEG_ZH = { high: '高頻', middle: '中頻', low: '低頻' };
  const ZONE_ZH = { common: '太常見', significant: '有效詞', rare: '太罕見' };
  const HOLD_IDS = ['rp-tiles', 'rp-freq', 'rp-power', 'rp-idf', 'rp-zones', 'rp-shift', 'rp-shift-table', 'rp-shift-notes', 'rp-top', 'rp-notes', 'zp-table', 'zp-scatter', 'cm-tiles', 'cm-plot', 'cm-lists', 'cm-notes'];
  // CF-DF map zones: colour, label, which list they feed
  const MAP_ZONES = { function: ['var(--s2)', '功能詞'], boilerplate: ['var(--s4)', '樣板詞'], topic: ['var(--s3)', '建庫依據詞'], keyword: ['var(--s1)', '文件關鍵詞'], number: ['var(--s5)', '數字殘渣'], other: ['var(--muted)', '其他'] };
  const MAP_THR = [['common_df', 'Function：DF/N ≥', 0.5, 1, 0.01], ['boiler_df', 'Boilerplate：DF/N ≥', 0.05, 0.6, 0.01], ['boiler_burst', 'Boilerplate：CF/DF <', 1, 2, 0.05],
    ['topic_df', 'Topic：DF/N ≥', 0.02, 0.5, 0.01], ['key_df', 'Keyword：DF/N <', 0.005, 0.2, 0.005], ['key_burst', 'Keyword：CF/DF ≥', 1.5, 6, 0.1]];
  const release = (...ids) => ids.forEach(id => { const e = $('#' + id); if (e) e.style.minHeight = ''; });

  async function loadZipf(force) {
    if (zs.data && !force) return;
    $('#zipf').innerHTML = '<div class="card">Computing the Zipf analysis…</div>';
    try {
      zs.data = await api('/api/zipf');
      zs.upper = zs.lower = null; zs.resAll = null;
      renderZipf();
      loadTerms();
      loadResolving();
      loadResolvingAll();
      loadMap();
    } catch (e) { $('#zipf').innerHTML = `<div class="card">${esc(e.message)}</div>`; }
  }

  function fitLine(c) {                                  // the regression line across the whole rank range
    const f = c.fit, y = r => 10 ** (f.intercept + f.slope * Math.log10(r));
    const floor = 0.6, rEnd = Math.min(c.vocabulary, 10 ** ((Math.log10(floor) - f.intercept) / f.slope));
    return [[1, y(1)], [rEnd, y(rEnd)]];
  }

  function renderZipf() {
    const z = zs.data, c = z.conditions.find(x => x.key === zs.cond), byKey = Object.fromEntries(z.conditions.map(x => [x.key, x]));
    if (!c.vocabulary) { $('#zipf').innerHTML = '<div class="card">The collection is empty – build one in <b>Add Documents</b>.</div>'; return; }
    const best = c.segments.slice().sort((a, b) => a.rmse - b.rmse)[0];
    const mid = c.segments.find(s => s.name === 'middle');
    const condBtns = z.conditions.map(x => `<button class="seg${x.key === zs.cond ? ' on' : ''}" data-cond="${x.key}" title="${esc(x.description)}">${x.key} · ${esc(x.name)}</button>`).join('');
    let h = `<div class="card help">
      <h2>Zipf's law — ${esc(z.collection)}</h2>
      <p>語料：<b>${fmt(c.documents)}</b> 篇 PubMed 英文摘要（標題 + 摘要，每篇以 PMID 為唯一 ID）。對每個詞計算 collection frequency <code>CF(t)</code>（全語料出現次數）與 document frequency <code>DF(t)</code>（出現的文件數），
        依 CF 由高到低排名後檢驗 <code>f(r) ∝ 1 / r<sup>k</sup></code>，即 <code>log f = log C − k · log r</code> 在 log-log 圖上是否為直線。</p>
      <div class="tiles">
        <div class="tile"><b>${fmt(c.documents)}</b><span>Documents</span></div>
        <div class="tile"><b>${fmt(c.tokens)}</b><span>Total tokens</span></div>
        <div class="tile"><b>${fmt(c.vocabulary)}</b><span>Unique terms (vocabulary)</span></div>
        <div class="tile"><b>${fmt(c.avg_tokens)}</b><span>Average tokens / document</span></div>
        <div class="tile"><b>${pct(c.hapax_share)}</b><span>of the vocabulary occurs once (hapax, ${fmt(c.hapax)} terms)</span></div>
        <div class="tile"><b>${pct(c.top10_share)}</b><span>of all tokens are the top-10 terms</span></div>
        <div class="tile accent"><b>${f2(c.fit.exponent)}</b><span>Zipf exponent k (whole curve)</span></div>
        <div class="tile accent"><b>${f2(c.fit.r2)}</b><span>R² of the log-log fit</span></div>
      </div>
      <h3 class="rp-h">Token vs term — A → E 一覽　<small class="muted">${fmt(c.documents)} 篇文件 · Δ 相對於箭頭所指的條件 · 點任一列即可切換</small></h3>
      <p><b>Token</b> 是文本切出來的每一個「出現」（occurrence）：句子 <code>the glp-1 receptor and the insulin receptor</code> 在條件 B 有 8 個 token。
        <b>Term</b>（unique term / type / vocabulary）是去重後不同的字串：同一句只有 5 個 term（<code>the</code>、<code>receptor</code> 各算一次）。
        Total tokens 加總的是前者、Unique terms 數的是後者；每個 term 的 CF 就是它的 token 數，所有 term 的 CF 加總 = total tokens。</p>
      <div class="scroll"><table class="grid num"><thead><tr><th>Condition</th><th>Total tokens</th><th>Δ</th><th>Unique terms</th><th>Δ</th><th>Tokens / term</th><th>Tokens / doc</th><th>Hapax</th><th>Top-10</th><th>Zipf k</th></tr></thead><tbody>
        ${z.conditions.map((x, i) => { const p = byKey[PREV[x.key]], d = (v, pv) => (p ? `<span class="${v > pv ? 'up' : v < pv ? 'down' : 'muted'}">${v > pv ? '+' : ''}${fmt(v - pv)}</span>` : '—');
          return `<tr class="click${x.key === zs.cond ? ' sel' : ''}" data-cond="${x.key}" title="${esc(x.description)}"><td><i class="key" style="background:${SERIES[i]}"></i><b>${x.key}</b> ${esc(x.name)}${p ? ` <small class="muted">← ${PREV[x.key]}</small>` : ''}</td><td><b>${fmt(x.tokens)}</b></td><td>${d(x.tokens, p && p.tokens)}</td><td><b>${fmt(x.vocabulary)}</b></td><td>${d(x.vocabulary, p && p.vocabulary)}</td><td>${f2(x.tokens / x.vocabulary, 1)}</td><td>${fmt(x.avg_tokens)}</td><td title="${fmt(x.hapax)} terms occur once">${pct(x.hapax_share)}</td><td title="share of tokens taken by the 10 most frequent terms">${pct(x.top10_share)}</td><td>${f2(x.fit.exponent)}</td></tr>`; }).join('')}
      </tbody></table></div>
      <div class="qa">
        <div><b>A → B</b>　token 反而<b>變多</b>（${fmt(byKey.A.tokens)} → ${fmt(byKey.B.tokens)}）、term 卻<b>大減</b>（${fmt(byKey.A.vocabulary)} → ${fmt(byKey.B.vocabulary)}）：拆掉標點把 <code>glp-1</code>、<code>p&lt;0.05</code> 這類黏著字串切成兩三個 token，同時 <code>receptor,</code>／<code>receptor.</code>／<code>(receptor</code> 併成同一個 term。</div>
        <div><b>B → C</b>　token <b>大減</b>（−${fmt(byKey.B.tokens - byKey.C.tokens)}，${pct((byKey.B.tokens - byKey.C.tokens) / byKey.B.tokens)}）、term 幾乎不變（−${fmt(byKey.B.vocabulary - byKey.C.vocabulary)}）：停用詞只有一百多個 term，但每個都出現成千上萬次。</div>
        <div><b>C → D</b>　token <b>完全不變</b>（stemming 不會刪字，只改寫）、term 減少 ${fmt(byKey.C.vocabulary - byKey.D.vocabulary)}：<code>receptor</code>／<code>receptors</code>、<code>treat</code>／<code>treated</code>／<code>treatment</code> 等併入同一個詞幹。</div>
        <div><b>B → E</b>　跳過停用詞、直接 stemming：token 跟 B 一樣（${fmt(byKey.E.tokens)}），term 由 ${fmt(byKey.B.vocabulary)} 降到 ${fmt(byKey.E.vocabulary)}；再和 D 比：E − D = ${fmt(byKey.E.vocabulary - byKey.D.vocabulary)} 個 term，就是停用詞經 stemming 後剩下的詞幹數（the / of / and 本身不會被合併，但 <code>was</code>／<code>were</code>／<code>be</code> 等仍各自成詞幹）。</div>
        <div><b>Tokens / term</b>　平均每個 term 出現幾次，是最直接的「型／例比」倒數：A ${f2(byKey.A.tokens / byKey.A.vocabulary, 1)} → D ${f2(byKey.D.tokens / byKey.D.vocabulary, 1)}，前處理的效果就是用更少的 term 承接同樣多（或更少）的 token。</div>
      </div>
    </div>`;

    // the condition switch is its own card, sticky under the page header, so it is reachable from anywhere in the tab
    h += `<div class="card cond-card"><div class="row"><b>Pre-processing condition</b><span class="segs">${condBtns}</span><span class="muted">${esc(c.description)}</span></div></div>`;

    h += `<div class="two">
      <div class="card"><h2>Experiment 1 — Rank vs frequency</h2>
        <div class="row"><span class="segs">
          <button class="seg${zs.scale === 'logy' ? ' on' : ''}" data-scale="logy">log frequency axis</button>
          <button class="seg${zs.scale === 'linear' ? ' on' : ''}" data-scale="linear">linear axes</button></span>
          <span class="muted">${zs.scale === 'linear' ? '線性座標下曲線貼著兩軸，幾乎看不出分布——這就是需要對數座標的原因。' : '頻率軸取對數後，整條長尾都看得見。'}</span></div>
        <div id="zp-rank"></div></div>
      <div class="card"><h2>Experiment 2 — Log-log plot + linear regression</h2>
        <div class="row"><span class="muted">log₁₀ CF = ${f2(c.fit.intercept)} − ${f2(c.fit.exponent)} · log₁₀ r　·　虛線為迴歸線，垂直虛線分隔高／中／低頻三段</span></div>
        <div id="zp-loglog"></div></div>
    </div>`;

    const fitRow = (name, f, extra = '') => `<tr><td>${name}</td><td>${fmt(f.from)} – ${fmt(f.to)}</td><td>${f2(f.slope)}</td><td>${f2(f.intercept)}</td><td><b>${f2(f.exponent)}</b></td><td>${f2(f.r2, 4)}</td><td>${f2(f.rmse, 4)}</td>${extra}</tr>`;
    h += `<div class="card"><h2>Regression report　<small class="muted">log₁₀(CF) = a − b · log₁₀(r) · condition ${c.key}</small></h2>
      <table class="grid num"><thead><tr><th>Range</th><th>Ranks</th><th>Slope (−b)</th><th>Intercept (a)</th><th>Zipf exponent k</th><th>R²</th><th>RMSE</th><th>Share of tokens</th><th>Mean residual vs whole-curve line</th></tr></thead><tbody>
      ${fitRow('<b>Whole curve</b>', c.fit, '<td>100%</td><td>0</td>')}
      ${c.segments.map(sg => fitRow(`${SEG_ZH[sg.name]} ${sg.name}-frequency terms`, sg, `<td>${pct(sg.tokens_share)}</td><td>${sg.global_bias > 0 ? '+' : ''}${f2(sg.global_bias)}</td>`)).join('')}
      </tbody></table>
      <p class="hint" style="margin-top:10px">三段是把 log(rank) 軸三等分。每段各自再做一次迴歸；最右欄是該段的點相對於「整條曲線迴歸線」的平均殘差（負值 = 實際頻率低於直線預測）。RMSE 的單位是 log₁₀ 頻率。</p>
      <div class="qa">
        <div><b>Q1 　Does the collection follow Zipf's law?</b> 近似符合，但不是嚴格符合。log-log 圖大致呈直線（R² = ${f2(c.fit.r2)}），中段幾乎是完美直線（R² = ${f2(mid.r2, 4)}）；
          但頭部比直線平（最高頻的詞頻率低於直線預測，平均殘差 ${f2(c.segments[0].global_bias)}），尾部則因頻率只能是整數 1、2、3… 而呈階梯狀並向下彎。</div>
        <div><b>Q2 　Estimated Zipf exponent?</b> 整條曲線 OLS：k = <b>${f2(c.fit.exponent)}</b>；只看中頻段：k = <b>${f2(mid.exponent)}</b>（較接近理想值 1）。
          整體估計偏高，是因為 ${fmt(c.segments[2].n)} 個低頻詞（佔詞彙 ${pct(c.segments[2].n / c.vocabulary)}）在未加權的 OLS 中佔了絕大多數資料點，把斜率拉向尾部較陡的斜率（k = ${f2(c.segments[2].exponent)}）。</div>
        <div><b>Q3 　Is R² sufficient to prove it?</b> 不足。任何遞減排序後的 rank–frequency 資料在 log-log 下 R² 都很高（排序本身保證單調）；R² 也看不出系統性偏差——本例 R² = ${f2(c.fit.r2)}，
          但三段各自的斜率是 ${c.segments.map(s => f2(s.exponent, 2)).join(' / ')}，相對整體迴歸線的平均殘差是 ${c.segments.map(s => (s.global_bias > 0 ? '+' : '') + f2(s.global_bias, 2)).join(' / ')}——殘差不是隨機散布而是隨 rank 有系統地變化，代表單一冪次律並不完全成立。還需要看殘差圖、分段斜率，或以 MLE／KS 檢定與 lognormal、Zipf–Mandelbrot 等替代模型比較。</div>
        <div><b>Q4 　Which portion fits best?</b> <b>${SEG_ZH[best.name]}（${best.name}-frequency）段</b>：RMSE = ${f2(best.rmse, 4)}、R² = ${f2(best.r2, 4)}。高頻段只有 ${c.segments[0].n} 個點且多為功能詞／主題詞，曲線較平；低頻段受整數頻率與 hapax（${pct(c.hapax_share)} 的詞只出現一次）影響呈階梯狀。</div>
      </div></div>`;

    h += `<div class="card"><h2>Resolving power of significant words　<small class="muted">Luhn (1958) · condition ${c.key}</small></h2>
      <p>Luhn 的觀察：排名最前的詞太常見、排名最後的詞太罕見，都分不開文件；鑑別力（resolving power）最高的「有效詞」落在 rank–frequency 曲線的中段，用 upper / lower 兩條截斷線框出來。
        這裡把每個詞的鑑別力量化為 <code>power(t) = CF(t) × idf(t)</code>，也就是該詞在全語料累積的 TF-IDF 權重：出現在幾乎每篇文件的詞 idf ≈ 0，只出現一兩次的詞 CF 太小，兩端都會被壓低。</p>
      <div class="row rp-ctl">
        <label>Upper cut-off（太常見）<input type="range" id="rp-upper" min="0" max="1000" step="1"><b id="rp-upper-v"></b></label>
        <label>Lower cut-off（太罕見）<input type="range" id="rp-lower" min="0" max="1000" step="1"><b id="rp-lower-v"></b></label>
        <button id="rp-auto" title="Cut-offs where the smoothed curve falls to half of its peak">Auto（半高寬）</button>
        <label><input type="checkbox" id="rp-nonum"${zs.nonum ? ' checked' : ''}>隱藏純數字詞</label>
      </div>
      <div id="rp-tiles" class="tiles"><span class="muted">Computing the resolving power…</span></div>
      <div class="two">
        <div><div id="rp-freq"></div><p class="hint">Rank–frequency 曲線（左軸，log）疊上鑑別力的移動中位數（右軸，線性）。著色區間是兩條截斷線之間的有效詞。</p></div>
        <div><div id="rp-power"></div><p class="hint">每個點是一個詞（前 400 名全畫，之後依 log 間距取樣）；實線是沿 rank 軸的移動中位數。滑鼠移到點上可看該詞。</p></div>
      </div>
      <div id="rp-zones" class="scroll" style="margin-top:6px"></div>
      <h3 class="rp-h">IDF 疊在 CF × rank 上　<small class="muted">鑑別力的兩個因子 · condition ${c.key}</small></h3>
      <div id="rp-idf"></div>
      <p class="hint">同一條 rank 軸：CF 走左軸（log），每個詞的 <code>IDF = log<sub>10</sub>(N / DF)</code> 走右軸（線性），實線是 IDF 的移動中位數。CF 沿 rank 單調下降、IDF 大致單調上升，兩者相乘的 <code>CF × IDF</code> 就是中間凸起的鑑別力曲線——有效詞是「CF 還夠大、IDF 也已經夠高」的交會區。</p>
      <h3 class="rp-h">A → E：有效詞分布怎麼移動　<small class="muted">各條件自己的 Auto 截斷線 · 粗線 / 表格反白 = 目前條件</small></h3>
      <div id="rp-shift"><span class="muted">Computing all conditions…</span></div>
      <p class="hint">細線是各條件的 CF（左軸，log），粗線是同一條件的鑑別力移動中位數（右軸）。圖上方的橫條是該條件的有效詞區間，圓點是鑑別力峰值所在的 rank。</p>
      <div id="rp-shift-table" class="scroll"></div>
      <div id="rp-shift-notes" class="qa"></div>
      <div class="two" style="margin-top:14px"><div><div id="rp-top" class="scroll"></div></div><div id="rp-notes" class="qa" style="margin-top:0"></div></div>
    </div>`;

    h += `<div class="card"><h2>Top 50 terms by collection frequency　<small class="muted">condition ${c.key}</small></h2>
      <div class="scroll"><table class="grid num"><thead><tr><th>Rank r</th><th>Term</th><th>CF</th><th>DF</th><th>CF / DF</th><th>% of tokens</th><th>r × CF</th></tr></thead><tbody>
      ${c.top.map(([t, cf, df], i) => `<tr><td>${i + 1}</td><td><code>${esc(t)}</code></td><td>${fmt(cf)}</td><td>${fmt(df)}</td><td>${f2(cf / df, 2)}</td><td>${pct(cf / c.tokens)}</td><td>${fmt((i + 1) * cf)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="hint" style="margin-top:8px">若 k = 1，<code>r × CF</code> 應近似常數。</p></div>`;

    h += `<div class="card"><h2>Effect of pre-processing — conditions A / B / C / D / E</h2>
      <table class="grid num"><thead><tr><th>Condition</th><th>Tokens</th><th>Vocabulary</th><th>Avg tokens / doc</th><th>Hapax share</th><th>Top-10 share</th><th>Zipf exponent k</th><th>R²</th><th>RMSE</th><th>k (middle)</th><th>Top 6 terms</th></tr></thead><tbody>
      ${z.conditions.map((x, i) => `<tr class="${x.key === zs.cond ? 'sel' : ''}"><td><i class="key" style="background:${SERIES[i]}"></i><b>${x.key}</b> ${esc(x.name)}</td><td>${fmt(x.tokens)}</td><td>${fmt(x.vocabulary)}</td><td>${fmt(x.avg_tokens)}</td><td>${pct(x.hapax_share)}</td><td>${pct(x.top10_share)}</td><td><b>${f2(x.fit.exponent)}</b></td><td>${f2(x.fit.r2)}</td><td>${f2(x.fit.rmse)}</td><td>${f2(x.segments[1].exponent)}</td><td class="t">${x.top.slice(0, 6).map(t => `<code>${esc(t[0])}</code>`).join(' ')}</td></tr>`).join('')}
      </tbody></table>
      <div class="two" style="margin-top:14px"><div id="zp-overlay"></div>
      <div class="qa">
        <div><b>Vocabulary size</b>　A→B：去除標點後，<code>data,</code>、<code>data.</code>、<code>(data</code> 合併為同一個詞，詞彙由 ${fmt(byKey.A.vocabulary)} 降到 ${fmt(byKey.B.vocabulary)}（−${pct(1 - byKey.B.vocabulary / byKey.A.vocabulary)}）；token 數反而增加，因為連字號與斜線把 <code>glp-1</code> 切成兩個 token。
          B→C：停用詞只有 ${fmt(byKey.B.vocabulary - byKey.C.vocabulary)} 個詞型，詞彙幾乎不變，但 token 少了 ${pct(1 - byKey.C.tokens / byKey.B.tokens)}。C→D：Porter stemming 把詞形變化合併，詞彙再降到 ${fmt(byKey.D.vocabulary)}（−${pct(1 - byKey.D.vocabulary / byKey.C.vocabulary)}）。</div>
        <div><b>High-frequency terms</b>　A、B 的榜首是 the / of / and 等功能詞（前 10 名佔全部 token 的 ${pct(byKey.B.top10_share)}）；移除停用詞後，榜首換成主題詞（glp、receptor、weight、obesity…），前 10 名只佔 ${pct(byKey.C.top10_share)}；stemming 後同一詞幹的變形頻率相加（patient + patients → patient），高頻詞的 CF 進一步升高。</div>
        <div><b>Zipf exponent</b>　整體 k：${z.conditions.map(x => `${x.key} = ${f2(x.fit.exponent, 2)}`).join('、')}。A 最接近 1，是因為黏著標點的詞製造了大量只出現一次的詞型，拉長了尾巴；去標點後尾巴變短、斜率變陡。移除停用詞砍掉曲線最高的頭部，頭段更平（高頻段 k：B = ${f2(byKey.B.segments[0].exponent, 2)} → C = ${f2(byKey.C.segments[0].exponent, 2)}）。Stemming 縮短尾巴並抬高中段，整體 k 上升。</div>
        <div><b>E = B + stemming（不去停用詞）</b>　E 的頭部與 B 相同（the / of / and 仍在榜首，前 10 名佔 ${pct(byKey.E.top10_share)}），但尾巴和 D 一樣被 stemming 縮短，詞彙 ${fmt(byKey.E.vocabulary)}、整體 k = ${f2(byKey.E.fit.exponent, 2)}（B ${f2(byKey.B.fit.exponent, 2)}、D ${f2(byKey.D.fit.exponent, 2)}）。把 E 和 D 疊在一起看，差別全在頭部，這正是停用詞對 Zipf 曲線的獨立效果；把 E 和 B 疊在一起看，差別全在尾部，是 stemming 的獨立效果。</div>
        <div><b>Shape</b>　五條曲線的中段近乎平行；差異集中在頭部（停用詞移除後變平、出現「肩膀」）與尾部（詞彙越小，曲線越早觸底）。前處理改變的是截距與頭尾，不改變「少數詞極常見、多數詞極罕見」的長尾本質。</div>
      </div></div></div>`;

    const st = z.stemming;
    h += `<div class="card"><h2>Porter stemming — what gets merged (C → D)</h2>
      <p>${fmt(st.words)} 個不同的詞被化為 ${fmt(st.stems)} 個詞幹（詞彙減少 <b>${pct(st.reduction)}</b>）；其中 ${fmt(st.merged_stems)} 個詞幹合併了兩個以上的詞形。以下是合併最多詞形的詞幹（括號內為各詞形的 CF）：</p>
      <div class="stem-groups">${st.groups.slice(0, 14).map(g => `<div class="stem-group"><span class="chip term"><b>${esc(g.stem)}</b> <small>CF ${fmt(g.cf)} · ${g.n_forms} forms</small></span>${g.forms.map(([w, n]) => `<span class="chip">${esc(w)} <small>${fmt(n)}</small></span>`).join('')}</div>`).join('')}</div></div>`;

    h += `<div class="card"><h2>CF vs DF vs IDF　<small class="muted">condition ${c.key} · idf(t) = log₁₀(N / df(t)), N = ${fmt(c.documents)}</small></h2>
      <div class="try"><input id="zp-terms" placeholder="terms to compare (space or comma separated) – leave empty for the default set" value="${esc(zs.terms)}"><button id="zp-terms-btn" class="primary">Compare</button></div>
      <div class="two"><div><div id="zp-table" class="scroll"></div></div><div><div id="zp-scatter"></div>
        <p class="hint">每個點是一個詞（CF 最高的 1,500 個）。對角線 CF = DF 代表「每篇最多出現一次」；離對角線越遠（越靠上）表示該詞集中在少數文件裡反覆出現（bursty）。圈起來的是左表的詞。</p></div></div>
      <div class="qa">
        <div><b>Why can a term have a high CF but a relatively low DF?</b> 因為它是少數文件的「主題詞」：只在少數文件出現，但在那些文件裡反覆出現（CF/DF 高，例如某個藥名或疾病名）。反之像 study、results 這類詞每篇出現一兩次，CF/DF 接近 1。</div>
        <div><b>Which measure tells whether a term is widely distributed?</b> DF（或 DF/N）。CF 只說總量，分不出「100 篇各 1 次」與「1 篇 100 次」。</div>
        <div><b>Why are both important in IR?</b> DF 決定 IDF（詞的鑑別力）與 posting list 的長度（查詢成本）；CF 決定位置索引的大小、語言模型的 collection probability（平滑用），且 CF 與 DF 的落差（burstiness）本身就是判斷「內容詞」的訊號。</div>
        <div><b>Why do very common terms get lower IDF? — Zipf 與 TF-IDF 的關係</b>　idf = log(N/df)：出現在幾乎每篇文件的詞 df ≈ N，idf ≈ 0，無法區分文件。Zipf 分布告訴我們頻率極度不均：排名最前的少數詞佔了大部分 token，若只用 TF 計分，分數會被這些詞主導；
          IDF 正好以 log 尺度抵銷這個冪次律的頭部，讓落在中、低頻段的內容詞得到較高權重——TF-IDF 等於是依 Zipf 曲線上的位置重新加權。</div>
      </div></div>`;

    h += `<div class="card"><h2>CF–DF map　<small class="muted">condition ${c.key} · 從 CF / DF 與 DF / N 把詞彙分區</small></h2>
      <p><code>CF / DF</code> 是「含該詞的文件裡平均出現幾次」（burstiness），<code>DF / N</code> 是「分布在多少比例的文件」。兩個比例把詞彙分成：<b>功能詞</b>（幾乎每篇都有）、<b>樣板詞</b>（分布廣但每篇只出現一次：results、methods）、<b>建庫依據詞</b>（分布中等又反覆出現：語料在講什麼）、<b>文件關鍵詞</b>（少數文件裡反覆出現）、<b>數字殘渣</b>。
        虛線是 Poisson 參考線 <code>DF* = N(1 − e<sup>−CF/N</sup>)</code>：詞若隨機散布，DF 會落在線上；點越在線的左邊（DF 遠小於 DF*），越是主題性的詞。</p>
      <div class="row rp-ctl" id="cm-ctl">${MAP_THR.map(([k, lab, lo, hi, st]) => `<label>${lab}<input type="range" data-thr="${k}" min="${lo}" max="${hi}" step="${st}"><b data-thr-v="${k}"></b></label>`).join('')}<button id="cm-reset">預設值</button></div>
      <div id="cm-tiles" class="tiles"><span class="muted">Computing the CF–DF map…</span></div>
      <div class="two"><div><div id="cm-plot"></div><p class="hint">每個點是一個詞（CF 最高的 1,500 個加上所有文件關鍵詞）。對角線 CF = DF：每篇最多出現一次。</p></div><div id="cm-lists"></div></div>
      <div id="cm-notes" class="qa"></div></div>`;

    // re-rendering must not make the page jump: the sections that are filled in asynchronously keep their previous
    // height (released once their new content is in) and the scroll position is restored
    const held = HOLD_IDS.map(id => [id, $('#' + id) ? $('#' + id).offsetHeight : 0]), y = window.scrollY;
    $('#zipf').innerHTML = h;
    held.forEach(([id, px]) => { const e = $('#' + id); if (e && px) e.style.minHeight = px + 'px'; });
    window.scrollTo(0, y);
    $$('#zipf [data-cond]').forEach(b => b.addEventListener('click', () => {
      zs.cond = b.dataset.cond; zs.upper = zs.lower = null; zs.res = null;
      renderZipf(); loadTerms(); loadResolving(); renderShift(); loadMap();
    }));
    $$('#zipf [data-scale]').forEach(b => b.addEventListener('click', () => { zs.scale = b.dataset.scale; renderZipf(); renderTerms(); renderResolving(); renderShift(); renderMap(); }));
    const toRank = v => sliderRank(v, c.vocabulary);
    const cutMoved = which => {
      zs.upper = toRank(+$('#rp-upper').value); zs.lower = toRank(+$('#rp-lower').value);
      if (zs.upper > zs.lower) { if (which === 'upper') zs.lower = zs.upper; else zs.upper = zs.lower; }
      $('#rp-upper-v').textContent = 'rank ' + fmt(zs.upper); $('#rp-lower-v').textContent = 'rank ' + fmt(zs.lower);
      clearTimeout(resTimer); resTimer = setTimeout(loadResolving, 120);
    };
    $('#rp-upper').addEventListener('input', () => cutMoved('upper'));
    $('#rp-lower').addEventListener('input', () => cutMoved('lower'));
    $('#rp-auto').addEventListener('click', () => { zs.upper = zs.lower = null; loadResolving(); });
    $('#rp-nonum').addEventListener('change', e => { zs.nonum = e.target.checked; renderResolving(); renderShift(); });
    $('#zp-terms-btn').addEventListener('click', () => { zs.terms = $('#zp-terms').value; loadTerms(); });
    $$('#cm-ctl [data-thr]').forEach(inp => inp.addEventListener('input', () => { zs.mapThr[inp.dataset.thr] = +inp.value; $(`#cm-ctl [data-thr-v=${inp.dataset.thr}]`).textContent = inp.value; loadMap(); }));
    $('#cm-reset').addEventListener('click', () => { zs.mapThr = {}; loadMap(); });
    renderMap();
    $('#zp-terms').addEventListener('keydown', e => { if (e.key === 'Enter') { zs.terms = $('#zp-terms').value; loadTerms(); } });

    const idx = z.conditions.indexOf(c), color = SERIES[idx];
    const pts = c.points.map(([r, f]) => [r, f, r <= c.top.length ? c.top[r - 1][0] : '']);
    plot($('#zp-rank'), { title: 'Rank vs frequency', series: [{ color, pts, mode: 'line' }], ylog: zs.scale === 'logy', xzero: true, yzero: true,
      xlabel: 'Rank r', ylabel: 'Collection frequency CF', tipX: v => 'rank ' + fmt(v), tipY: v => 'CF ' + fmt(v) });
    plot($('#zp-loglog'), { title: 'Log-log rank-frequency plot', xlog: true, ylog: true, xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)',
      series: [{ name: 'Observed CF', color, pts, mode: 'line' }, { name: `Regression line (k = ${f2(c.fit.exponent)})`, color: 'var(--ink)', pts: fitLine(c), mode: 'fit', tip: false }],
      vlines: c.segments.slice(1).map(sg => ({ x: sg.from })), bands: c.segments.map(sg => ({ from: sg.from, to: sg.to, label: sg.name })),
      tipX: v => 'rank ' + fmt(v), tipY: v => 'CF ' + fmt(Math.round(v)) });
    plot($('#zp-overlay'), { title: 'Rank-frequency curves of the conditions', xlog: true, ylog: true, xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)', height: 360,
      series: z.conditions.map((x, i) => ({ name: `${x.key} · ${x.name}`, color: SERIES[i], pts: x.points, mode: 'line' })),
      tipX: v => 'near rank ' + fmt(v), tipY: v => 'CF ' + fmt(v) });
  }

  /* ---- resolving power of significant words (Luhn) ---- */
  let resTimer = null, resSeq = 0;
  const sliderRank = (v, V) => Math.max(1, Math.min(V, Math.round(V ** (v / 1000))));      // the sliders are on a log-rank scale
  async function loadResolving() {
    const seq = ++resSeq, q = { cond: zs.cond };
    if (zs.upper) q.upper = zs.upper;
    if (zs.lower) q.lower = zs.lower;
    try {
      const r = await api('/api/zipf/resolving?' + new URLSearchParams(q));
      if (seq !== resSeq) return;                        // a newer request is on its way
      zs.res = r;
      renderResolving();
    } catch (e) { const t = $('#rp-tiles'); if (t) t.textContent = e.message; }
  }
  function renderResolving() {
    const r = zs.res, box = $('#rp-tiles');
    if (!r || !box || r.condition !== zs.cond || !r.vocabulary) return;
    const c = zs.data.conditions.find(x => x.key === zs.cond), V = r.vocabulary;
    const [common, sig, rare] = r.zones, isAuto = r.upper === r.auto.upper && r.lower === r.auto.lower;
    const toSlider = rank => Math.round(1000 * Math.log(rank) / Math.log(Math.max(V, 2)));
    [['#rp-upper', r.upper], ['#rp-lower', r.lower]].forEach(([id, rank]) => {             // leave a slider alone while it already shows this rank
      if (sliderRank(+$(id).value, V) !== rank) $(id).value = toSlider(rank);
    });
    $('#rp-upper-v').textContent = 'rank ' + fmt(r.upper); $('#rp-lower-v').textContent = 'rank ' + fmt(r.lower);
    $('#rp-auto').disabled = isAuto;
    const isNum = t => /^\d+$/.test(t), keep = t => !zs.nonum || !isNum(t);

    box.innerHTML = `
      <div class="tile accent"><b>${fmt(sig.terms)}</b><span>significant words · rank ${fmt(sig.from)} – ${fmt(sig.to)}（詞彙的 ${pct(sig.vocab_share)}）</span></div>
      <div class="tile accent"><b>${pct(sig.power_share)}</b><span>of the total resolving power</span></div>
      <div class="tile"><b>${pct(sig.tokens_share)}</b><span>of all tokens are significant words</span></div>
      <div class="tile"><b>${f2(sig.mean_power, 1)}</b><span>mean power per significant word（太罕見區 ${f2(rare.mean_power, 1)}）</span></div>
      <div class="tile"><b>${fmt(common.terms)}</b><span>terms above the upper cut-off · ${pct(common.tokens_share)} of tokens</span></div>
      <div class="tile"><b>${fmt(rare.terms)}</b><span>terms below the lower cut-off · ${pct(rare.tokens_share)} of tokens</span></div>`;

    const color = SERIES[zs.data.conditions.indexOf(c)], band = [{ from: r.upper, to: r.lower, color: 'var(--s3)' }];
    const cuts = [{ x: r.upper }, { x: r.lower }];
    const bands = r.zones.filter(z => z.terms).map(z => ({ from: z.from, to: z.to, label: ZONE_ZH[z.name] }));
    plot($('#rp-freq'), { title: 'Rank-frequency curve with the upper and lower cut-offs', xlog: true, ylog: true, xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)',
      series: [{ name: 'CF (left axis)', color, pts: c.points, mode: 'line' },
        { name: 'Resolving power, running median (right axis)', color: 'var(--s3)', pts: r.curve, mode: 'line', axis: 'r', width: 2.6, tipY: v => 'power ' + f2(v, 1) }],
      y2: { zero: true, label: 'Resolving power CF × IDF' }, shade: band, vlines: cuts, bands,
      tipX: v => 'rank ' + fmt(v), tipY: v => 'CF ' + fmt(v) });

    // points: [rank, power, label, emphasised, cf, df, term]
    const inBand = p => p[0] >= r.upper && p[0] <= r.lower;
    const labelled = new Set(r.top.filter(t => keep(t.term)).slice(0, 14).map(t => t.term));
    const pts = r.points.filter(p => keep(p[2])).map(([rank, power, term, cf, df]) => [rank, power, labelled.has(term) ? term : '', false, cf, df, term]);
    plot($('#rp-power'), { title: 'Resolving power of each term against its rank', xlog: true, yzero: true, xlabel: 'Rank r (log scale)', ylabel: 'Resolving power CF × IDF', hover: 'nearest', labels: true,
      series: [{ name: 'Outside the cut-offs', color: 'var(--muted)', mode: 'dots', r: 2.5, opacity: 0.4, pts: pts.filter(p => !inBand(p)).map(p => [p[0], p[1], '', false, p[4], p[5], p[6]]) },
        { name: 'Significant words', color: 'var(--s3)', mode: 'dots', r: 3.2, opacity: 0.85, pts: pts.filter(inBand) },
        { name: 'Running median', color: 'var(--ink)', mode: 'line', pts: r.curve }],
      shade: band, vlines: cuts,
      tipRows: (p, se) => (p.length < 7 ? [[se.color, 'running median', ''], [null, f2(p[1], 1), 'around rank ' + fmt(p[0])]]
        : [[se.color, String(p[6]), ''], [null, f2(p[1], 1), 'CF × IDF'], [null, fmt(p[0]), 'rank'], [null, fmt(p[4]), 'CF'], [null, fmt(p[5]), 'DF'], [null, f2(Math.log10(r.documents / p[5])), 'IDF']]) });

    // IDF against the same rank axis: [rank, idf, label, emphasised, cf, df, term]
    const idfOf = df => Math.log10(r.documents / df), maxIdf = Math.log10(r.documents);
    const idfPts = pts.map(p => [p[0], idfOf(p[5]), p[2], false, p[4], p[5], p[6]]);
    const idfTip = (p, se) => (p.length < 7 ? [[se.color, 'IDF running median', ''], [null, f2(p[1]), 'around rank ' + fmt(p[0])]]
      : [[se.color, String(p[6]), ''], [null, f2(p[1]), 'IDF'], [null, fmt(p[0]), 'rank'], [null, fmt(p[4]), 'CF'], [null, fmt(p[5]), 'DF of ' + fmt(r.documents)], [null, f2(p[4] * p[1], 1), 'CF × IDF']]);
    plot($('#rp-idf'), { title: 'Rank-frequency curve with the IDF of each term', width: 1100, height: 400, xlog: true, ylog: true,
      xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)', y2: { zero: true, label: 'IDF = log10(N / DF)' }, hover: 'nearest', labels: true,
      series: [{ name: 'CF (left axis)', color, pts: c.points.map(p => [p[0], p[1]]), mode: 'line', width: 2.2, opacity: 0.8 },
        { name: 'IDF, outside the cut-offs', color: 'var(--muted)', mode: 'dots', r: 2.2, opacity: 0.35, axis: 'r', pts: idfPts.filter(p => !inBand(p)).map(p => [p[0], p[1], '', false, p[4], p[5], p[6]]) },
        { name: 'IDF, significant words', color: 'var(--s2)', mode: 'dots', r: 3, opacity: 0.8, axis: 'r', pts: idfPts.filter(inBand) },
        { name: 'IDF running median (right axis)', color: 'var(--ink)', mode: 'line', axis: 'r', width: 2.4, pts: r.idf_curve },
        { name: 'IDF ceiling: DF = 1', legend: false, color: 'var(--muted)', mode: 'fit', axis: 'r', pts: [[1, maxIdf], [V, maxIdf]] }],
      shade: band, vlines: cuts, bands,
      tipRows: (p, se) => (se.axis === 'r' ? idfTip(p, se) : [[se.color, 'CF', ''], [null, fmt(p[1]), 'at rank ' + fmt(p[0])]]) });

    $('#rp-zones').innerHTML = `<table class="grid num"><thead><tr><th>Zone</th><th>Ranks</th><th>Terms</th><th>% of vocabulary</th><th>% of tokens</th><th>% of power</th><th>Mean power</th><th>Mean IDF</th><th>Top-ranked terms</th></tr></thead><tbody>` +
      r.zones.map(z => `<tr class="${z.name === 'significant' ? 'sel' : ''}"><td><b>${ZONE_ZH[z.name]}</b> ${z.name}</td><td>${z.terms ? fmt(z.from) + ' – ' + fmt(z.to) : '—'}</td><td>${fmt(z.terms)}</td><td>${pct(z.vocab_share)}</td><td>${pct(z.tokens_share)}</td><td>${pct(z.power_share)}</td><td>${f2(z.mean_power, 1)}</td><td>${f2(z.mean_idf)}</td><td class="t">${z.examples.slice(0, 6).map(t => `<code>${esc(t)}</code>`).join(' ') || '—'}</td></tr>`).join('') +
      '</tbody></table>';

    const top = r.top.filter(t => keep(t.term)).slice(0, 50), maxP = Math.max(...top.map(t => t.power), 0.001);
    $('#rp-top').innerHTML = `<table class="grid num"><thead><tr><th>Significant word</th><th>Rank</th><th>CF</th><th>DF</th><th>IDF</th><th>CF × IDF</th><th></th></tr></thead><tbody>` +
      top.map(t => `<tr><td><code>${esc(t.term)}</code></td><td>${fmt(t.rank)}</td><td>${fmt(t.cf)}</td><td>${fmt(t.df)}</td><td>${f2(t.idf)}</td><td><b>${f2(t.power, 1)}</b></td><td class="barcell"><i style="width:${Math.max(1, t.power / maxP * 100)}%;background:var(--s3)"></i></td></tr>`).join('') +
      '</tbody></table>';

    const stopNote = common.terms
      ? `目前 upper cut-off 以上有 ${fmt(common.terms)} 個詞（${common.examples.slice(0, 8).map(t => `<code>${esc(t)}</code>`).join(' ')}…），其中 ${fmt(common.stopwords)} 個在本系統的停用詞表裡，它們佔了 ${pct(common.tokens_share)} 的 token，卻只貢獻 ${pct(common.power_share)} 的鑑別力。`
      : `目前 upper cut-off 在 rank 1，沒有任何詞被當成「太常見」${'CD'.includes(r.condition) ? '——條件 ' + r.condition + ' 已經先移除停用詞，等於事先做過 upper cut-off；切到條件 B 可以看到曲線左端被停用詞壓低的樣子。' : '。'}`;
    $('#rp-notes').innerHTML = `
      <div><b>怎麼讀這兩張圖</b>　左圖是 Zipf 曲線加上兩條截斷線；右圖把同一條 rank 軸上每個詞的 <code>CF × IDF</code> 畫出來。移動中位數的峰值在 rank ${fmt(r.peak.rank)} 附近（${f2(r.peak.power, 1)}）。
        Auto 取的是曲線維持在峰值一半以上的區間：rank ${fmt(r.auto.upper)} – ${fmt(r.auto.lower)}${isAuto ? '（目前套用中）' : '；目前是手動設定'}。</div>
      <div><b>有效詞佔多少</b>　rank ${fmt(sig.from)} – ${fmt(sig.to)} 只有 ${fmt(sig.terms)} 個詞（詞彙的 ${pct(sig.vocab_share)}），平均每個詞的鑑別力是 ${f2(sig.mean_power, 1)}，是太罕見區（${f2(rare.mean_power, 1)}）的 ${rare.mean_power ? f2(sig.mean_power / rare.mean_power, 1) : '—'} 倍。</div>
      <div><b>Upper cut-off 與停用詞</b>　${stopNote}</div>
      <div><b>IDF 圖怎麼看</b>　三區的平均 IDF 是 ${f2(common.mean_idf)} → ${f2(sig.mean_idf)} → ${f2(rare.mean_idf)}（上限 log<sub>10</sub>(${fmt(r.documents)}) = ${f2(Math.log10(r.documents))}，即 DF = 1）。
        太常見區的 IDF 貼近 0，把再大的 CF 都乘掉；太罕見區的 IDF 已經頂到上限，但 CF 只剩個位數；只有中段兩個因子都不小，所以 <code>CF × IDF</code> 在那裡凸起。</div>
      <div><b>Lower cut-off 的代價</b>　截斷線以下有 ${fmt(rare.terms)} 個詞（詞彙的 ${pct(rare.vocab_share)}）。單看每個詞，鑑別力很低（平均 IDF 高達 ${f2(rare.mean_idf, 2)}，但 CF 太小）；但加總起來仍佔全部鑑別力的 ${pct(rare.power_share)}。
        所以 Luhn 的 lower cut-off 適合用來挑「代表語料主題的詞」（摘要、關鍵詞），檢索系統則通常保留長尾，改用 IDF 加權。</div>`;
    release('rp-tiles', 'rp-freq', 'rp-power', 'rp-idf', 'rp-zones', 'rp-top', 'rp-notes');
  }

  async function loadResolvingAll() {
    try {
      const all = await Promise.all(zs.data.conditions.map(x => api('/api/zipf/resolving?' + new URLSearchParams({ cond: x.key }))));
      zs.resAll = Object.fromEntries(all.map(r => [r.condition, r]));
      renderShift();
    } catch (e) { const t = $('#rp-shift'); if (t) t.textContent = e.message; }
  }
  function renderShift() {
    const all = zs.resAll, box = $('#rp-shift');
    if (!all || !box) return;
    const conds = zs.data.conditions.filter(x => all[x.key] && all[x.key].vocabulary), R = k => all[k];
    if (!conds.length) return;
    const colorOf = x => SERIES[zs.data.conditions.indexOf(x)], keep = t => !zs.nonum || !/^\d+$/.test(t);
    plot(box, { title: 'Rank-frequency curves and resolving power of the conditions', width: 1100, height: 400, xlog: true, ylog: true,
      xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)', y2: { zero: true, label: 'Resolving power (running median)' },
      series: conds.map(x => ({ name: `${x.key} CF`, legend: false, color: colorOf(x), pts: x.points, mode: 'line', width: 1.2, opacity: 0.55, tipY: v => 'CF ' + fmt(v) }))
        .concat(conds.map(x => ({ name: `${x.key} · ${x.name}`, color: colorOf(x), pts: R(x.key).curve, mode: 'line', axis: 'r', width: x.key === zs.cond ? 3.4 : 2.2, tipY: v => 'power ' + f2(v, 1) }))),
      strips: conds.map(x => ({ from: R(x.key).auto.upper, to: R(x.key).auto.lower, color: colorOf(x), label: x.key, mark: R(x.key).peak.rank })),
      tipX: v => 'near rank ' + fmt(v) });

    $('#rp-shift-table').innerHTML = `<table class="grid num"><thead><tr><th>Condition</th><th>Significant ranks</th><th>Words</th><th>% of tokens</th><th>% of power</th><th>Peak rank</th><th>Too common</th><th>Most discriminating significant words</th></tr></thead><tbody>` +
      conds.map(x => { const r = R(x.key), [common, sig] = r.zones;
        return `<tr class="${x.key === zs.cond ? 'sel' : ''}"><td><i class="key" style="background:${colorOf(x)}"></i><b>${x.key}</b> ${esc(x.name)}</td><td>${fmt(sig.from)} – ${fmt(sig.to)}</td><td>${fmt(sig.terms)}</td><td>${pct(sig.tokens_share)}</td><td>${pct(sig.power_share)}</td><td>${fmt(r.peak.rank)}</td><td>${fmt(common.terms)}</td><td class="t">${r.top.filter(t => keep(t.term)).slice(0, 7).map(t => `<code>${esc(t.term)}</code>`).join(' ')}</td></tr>`; }).join('') +
      '</tbody></table>';

    const by = Object.fromEntries(conds.map(x => [x.key, R(x.key)])), span = k => `rank ${fmt(by[k].auto.upper)} – ${fmt(by[k].auto.lower)}`;
    release('rp-shift', 'rp-shift-table', 'rp-shift-notes');
    if (!(by.A && by.B && by.C && by.D)) { $('#rp-shift-notes').innerHTML = ''; return; }
    $('#rp-shift-notes').innerHTML = `
      <div><b>A → B（去標點）</b>　有效詞區間由 ${span('A')} 變成 ${span('B')}。黏著標點的詞型（<code>data,</code>、<code>data.</code>）合併後，同一個詞的 CF 集中，鑑別力峰值由 ${f2(by.A.peak.power, 1)} 變為 ${f2(by.B.peak.power, 1)}；區間以上有 ${fmt(by.B.zones[0].terms)} 個太常見的詞。</div>
      <div><b>B → C（去停用詞）</b>　區間變成 ${span('C')}，upper cut-off 以上的詞由 ${fmt(by.B.zones[0].terms)} 個變為 ${fmt(by.C.zones[0].terms)} 個。停用詞表做的事等於事先套用了 upper cut-off：曲線最左端被 idf ≈ 0 壓低的那一段先被拿掉了。</div>
      <div><b>C → D（Porter stemming）</b>　區間變成 ${span('D')}。同詞幹的變形頻率相加，詞彙由 ${fmt(by.C.vocabulary)} 降到 ${fmt(by.D.vocabulary)}，有效詞佔 token 的比例由 ${pct(by.C.zones[1].tokens_share)} 變為 ${pct(by.D.zones[1].tokens_share)}、佔總鑑別力由 ${pct(by.C.zones[1].power_share)} 變為 ${pct(by.D.zones[1].power_share)}。</div>
      ${by.E ? `<div><b>B → E（不去停用詞、直接 stemming）</b>　區間變成 ${span('E')}，upper cut-off 以上仍有 ${fmt(by.E.zones[0].terms)} 個太常見的詞（D 是 ${fmt(by.D.zones[0].terms)} 個）：stemming 不會動到 the / of / and，所以曲線左端被 idf ≈ 0 壓低的那段還在；有效詞佔總鑑別力 ${pct(by.E.zones[1].power_share)}（B ${pct(by.B.zones[1].power_share)}、D ${pct(by.D.zones[1].power_share)}）。</div>` : ''}
      <div><b>注意 rank 不能直接跨條件對照</b>　每個條件都重新排名：C、D 少了停用詞，同一個詞的 rank 會比在 B 裡靠前。要比較的是區間相對於各自曲線的位置與寬度，而不是 rank 數字本身。</div>`;
  }

  async function loadTerms() {
    try {
      zs.table = await api('/api/zipf/terms?' + new URLSearchParams({ cond: zs.cond, terms: zs.terms, scatter: 1 }));
      renderTerms();
    } catch (e) { const t = $('#zp-table'); if (t) t.textContent = e.message; }
  }
  function renderTerms() {
    const t = zs.table, box = $('#zp-table');
    if (!t || !box) return;
    const maxIdf = Math.max(...t.rows.map(r => r.idf), 0.001);
    box.innerHTML = `<table class="grid num"><thead><tr><th>Term</th><th>CF</th><th>DF</th><th>CF / DF</th><th>DF / N</th><th>IDF</th><th></th></tr></thead><tbody>` +
      t.rows.slice().sort((a, b) => b.cf - a.cf).map(r => `<tr><td><code>${esc(r.term)}</code></td><td>${fmt(r.cf)}</td><td>${fmt(r.df)}</td><td>${f2(r.cf_per_doc, 2)}</td><td>${pct(r.df_ratio)}</td><td><b>${f2(r.idf)}</b></td><td class="barcell"><i style="width:${Math.max(1, r.idf / maxIdf * 100)}%"></i></td></tr>`).join('') +
      '</tbody></table>' + (t.missing.length ? `<p class="hint">Not in the vocabulary under condition ${esc(t.condition)} (removed as stop words or never occur): ${t.missing.map(w => `<code>${esc(w)}</code>`).join(' ')}</p>` : '');
    const chosen = new Set(t.rows.map(r => r.term));
    const others = t.scatter.filter(p => !chosen.has(p[0])).map(([w, cf, df]) => [df, cf, w]);
    const picked = t.rows.map(r => [r.df, r.cf, r.term]);
    plot($('#zp-scatter'), { title: 'CF vs DF', xlog: true, ylog: true, xlabel: 'Document frequency DF (log scale)', ylabel: 'Collection frequency CF (log scale)', height: 380, hover: 'nearest', labels: true,
      series: [{ name: 'CF = DF', color: 'var(--muted)', mode: 'fit', pts: [[1, 1], [t.documents, t.documents]], tip: false },
        { name: 'Other terms', color: 'var(--s1)', mode: 'dots', r: 2.5, opacity: 0.35, pts: others.map(p => [p[0], p[1]]).map((p, i) => [p[0], p[1], '', false, others[i][2]]) },
        { name: 'Compared terms', color: 'var(--s2)', mode: 'dots', r: 4.5, opacity: 1, pts: picked }],
      tipRows: (p, se) => [[se.color === 'var(--muted)' ? null : se.color, String(p[2] || p[4] || ''), ''], [null, fmt(p[1]), 'CF'], [null, fmt(p[0]), 'DF'], [null, f2(p[1] / p[0], 2), 'CF / DF'], [null, f2(Math.log10(t.documents / p[0])), 'IDF']] });
    release('zp-table', 'zp-scatter');
  }

  // ---- CF-DF map ------------------------------------------------------- //
  let mapTimer = null, mapSeq = 0;
  function loadMap() {
    clearTimeout(mapTimer);
    mapTimer = setTimeout(async () => {
      const seq = ++mapSeq;
      try {
        const r = await api('/api/zipf/cfdf?' + new URLSearchParams({ cond: zs.cond, ...zs.mapThr }));
        if (seq !== mapSeq) return;
        zs.map = r; renderMap();
      } catch (e) { const t = $('#cm-tiles'); if (t) t.textContent = e.message; }
    }, 150);
  }
  function renderMap() {
    const r = zs.map, box = $('#cm-tiles');
    if (!r || !box || r.condition !== zs.cond || !r.vocabulary) return;
    const thr = r.thresholds, N = r.documents, Z = Object.fromEntries(r.zones.map(z => [z.name, z]));
    MAP_THR.forEach(([k]) => { const inp = $(`#cm-ctl [data-thr=${k}]`); if (inp && +inp.value !== thr[k]) inp.value = thr[k]; $(`#cm-ctl [data-thr-v=${k}]`).textContent = thr[k]; });
    box.innerHTML = r.zones.map(z => `<div class="tile${z.name === 'other' ? '' : ' accent'}"><b><i class="key" style="background:${MAP_ZONES[z.name][0]}"></i>${fmt(z.terms)}</b><span>${MAP_ZONES[z.name][1]} ${z.name} · ${pct(z.tokens_share)} of tokens · mean IDF ${f2(z.mean_idf, 2)}</span></div>`).join('');

    // points: [df, cf, label, emphasised, term, zone, burst, poisson ratio]
    const pt = ([t, cf, df, z]) => [df, cf, '', false, t, z, cf / df, N * (1 - Math.exp(-cf / N)) / df];
    const byZone = Object.fromEntries(Object.keys(MAP_ZONES).map(z => [z, []]));
    r.points.forEach(p => byZone[p[3]].push(pt(p)));
    plot($('#cm-plot'), { title: 'CF-DF map', xlog: true, ylog: true, xlabel: 'Document frequency DF (log scale)', ylabel: 'Collection frequency CF (log scale)', height: 400, hover: 'nearest',
      series: [{ name: 'CF = DF', color: 'var(--line)', mode: 'line', pts: [[1, 1], [N, N]], tip: false, legend: false, width: 1 },
        { name: 'Poisson DF* (random scatter)', color: 'var(--ink)', mode: 'fit', pts: r.poisson.map(([cf, df]) => [df, cf]), tip: false }]
        .concat(['other', 'number', 'keyword', 'topic', 'boilerplate', 'function'].map(z => ({ name: `${MAP_ZONES[z][1]} ${z}`, color: MAP_ZONES[z][0], mode: 'dots', r: z === 'other' ? 2.2 : 3.2, opacity: z === 'other' ? 0.3 : 0.85, pts: byZone[z] }))),
      tipRows: (p, se) => (p.length < 5 ? [[se.color, se.name, '']] : [[se.color, String(p[4]), MAP_ZONES[p[5]][1]], [null, fmt(p[1]), 'CF'], [null, fmt(p[0]), 'DF'], [null, f2(p[6], 2), 'CF / DF'], [null, pct(p[0] / N), 'DF / N'], [null, f2(p[7], 2) + '×', 'burstier than random (DF* / DF)'], [null, f2(Math.log10(N / p[0])), 'IDF']]) });

    const chips = (names, n, title, why) => { const xs = names.flatMap(z => Z[z].examples).slice(0, n); return `<div class="cm-list"><b>${title}</b> <small class="muted">${why}</small><div>${xs.map(x => `<span class="chip" title="CF ${fmt(x.cf)} · DF ${fmt(x.df)} · CF/DF ${f2(x.burst, 2)} · ${f2(x.poisson, 2)}× burstier than random · IDF ${f2(x.idf)}">${esc(x.term)} <small>${fmt(x.cf)}/${fmt(x.df)}</small></span>`).join('') || '<span class="muted">—</span>'}</div></div>`; };
    $('#cm-lists').innerHTML =
      chips(['function', 'boilerplate'], 24, '潛在移除詞（領域停用詞候選）', `功能詞 ${fmt(Z.function.terms)} + 樣板詞 ${fmt(Z.boilerplate.terms)}：到處都有，IDF ≈ 0 或每篇只出現一次`) +
      chips(['number'], 18, '數字殘渣', `${fmt(Z.number.terms)} 個純數字 token，佔 ${pct(Z.number.tokens_share)} 的 token`) +
      chips(['topic'], 24, '建庫依據詞', `${fmt(Z.topic.terms)} 個：分布中等又反覆出現，描述語料主題`) +
      chips(['keyword'], 24, '文件關鍵詞', `${fmt(Z.keyword.terms)} 個：集中在少數文件裡反覆出現，TF-IDF 高`);
    const stopLike = Z.topic.examples.filter(x => zs.data && /^(were|was|is|are|be|been|this|that|these|those|as|or|on|by|from|at|an|it|its|we|our|not|than|which|has|have|had|also|may|can|both|between|after|among|more|most|other|such|all|their|its)$/.test(x.term)).map(x => x.term);
    $('#cm-notes').innerHTML = `
      <div><b>IDF 就在 x 軸上</b>　idf = log<sub>10</sub>(N / DF)：越靠左 IDF 越高。功能詞與樣板詞都擠在最右邊，IDF 不需要停用詞表就把它們一起壓平——功能詞區的平均 IDF 只有 ${f2(Z.function.mean_idf, 3)}。</div>
      <div><b>TF 是離對角線的距離</b>　CF / DF 越大，該詞在含它的文件裡 tf 越高。文件關鍵詞兩個因子都大（平均 IDF ${f2(Z.keyword.mean_idf, 2)}、CF/DF ≥ ${thr.key_burst}），是 TF-IDF 最高的一群，適合自動關鍵詞、文件摘要與分群特徵。</div>
      <div><b>樣板詞是 Snowball 表沒有的停用詞</b>　${Z.boilerplate.examples.slice(0, 6).map(x => `<code>${esc(x.term)}</code>`).join(' ')} 每篇恰好出現一次（CF/DF ≈ 1）、貼著 Poisson 線：它們是結構化摘要的小標題，對檢索沒有鑑別力，是領域停用詞表的第一批候選。</div>
      <div><b>這張圖分不出的東西</b>　${stopLike.length ? `建庫依據詞區裡混著 ${stopLike.slice(0, 6).map(t => `<code>${esc(t)}</code>`).join(' ')}：它們的 CF/DF 與 DF/N 和 weight、patients 幾乎一樣，` : '功能詞與主題詞的 CF/DF、DF/N 可以長得一樣，'}純靠統計量分不出「功能詞」與「主題詞」，這正是停用詞表提供的語言知識。切到條件 C 可以看到去掉停用詞後這一區只剩內容詞。</div>`;
    release('cm-tiles', 'cm-plot', 'cm-lists', 'cm-notes');
  }

  /* ================================================================== *
   * word2vec tab
   * ================================================================== */
  let w2vTimer = null, w2vShown = '';        // w2vShown: which trained model the result panels belong to
  async function w2vStatus() {
    let s;
    try { s = await api('/api/w2v/status'); } catch (e) { $('#w2v-info').textContent = e.message; return; }
    const job = s.job, prog = $('#w2v-progress');
    $('#w2v-train').disabled = job.state === 'running';
    if (job.state === 'running') {
      prog.classList.remove('hidden');
      $('i', prog).style.width = (job.progress * 100).toFixed(1) + '%';
      $('span', prog).textContent = `Training ${job.params.model} on ${fmt(job.documents)} documents… ${(job.progress * 100).toFixed(1)}% · epoch ${job.epoch || 1}/${job.params.epochs}` +
        (job.loss != null ? ` · loss ${job.loss} · learning rate ${job.alpha}` : '') + ` · ${job.elapsed}s`;
      clearTimeout(w2vTimer); w2vTimer = setTimeout(w2vStatus, 1000);
    } else {
      prog.classList.toggle('hidden', job.state !== 'error');
      if (job.state === 'error') $('span', prog).textContent = 'Training failed: ' + job.message;
    }
    const m = s.model;
    if (!m) {
      $('#w2v-info').innerHTML = `<p class="muted">No model yet. Training corpus: ${esc(s.collection)} – ${fmt(s.documents)} documents. Press <b>Train</b> (pure Python: skip-gram needs roughly 25 s per epoch for 1,000 abstracts, CBOW about 8 s).</p>`;
      return;
    }
    $('#w2v-info').innerHTML = `<div class="tiles">
        <div class="tile accent"><b>${esc(m.model)}</b><span>Model · negative sampling (${m.negative})</span></div>
        <div class="tile"><b>${fmt(m.vocabulary)}</b><span>Vocabulary (count ≥ ${m.min_count})</span></div>
        <div class="tile"><b>${m.dim}</b><span>Vector dimensions</span></div>
        <div class="tile"><b>±${m.window}</b><span>Window (dynamic)</span></div>
        <div class="tile"><b>${fmt(m.documents || 0)}</b><span>Documents</span></div>
        <div class="tile"><b>${fmt(m.sentences || 0)}</b><span>Sentences</span></div>
        <div class="tile"><b>${fmt(m.corpus_words)}</b><span>Training words</span></div>
        <div class="tile"><b>${m.epochs}</b><span>Epochs · ${fmt(m.train_seconds)} s</span></div>
      </div>
      <div class="two"><div><h4>Training loss per epoch</h4><div id="w2v-loss"></div></div>
      <div><h4>Most frequent words in the model</h4><div class="chips">${m.top_words.map(([w, n]) => `<span class="chip term"><a href="#" data-w="${esc(w)}">${esc(w)}</a> <small>${fmt(n)}</small></span>`).join('')}</div></div></div>`;
    plot($('#w2v-loss'), { title: 'Training loss', height: 220, xlabel: 'Epoch', ylabel: 'Mean loss per training pair',
      series: [{ color: 'var(--s1)', mode: 'line', pts: m.loss_history.map((l, i) => [i + 1, l]) }], tipX: v => 'epoch ' + v, tipY: v => 'loss ' + v });
    $$('#w2v-info a[data-w]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#w2v-word').value = a.dataset.w; w2vSimilar(); }));
    const key = [m.model, m.dim, m.window, m.epochs, m.train_seconds, m.vocabulary].join('|');
    if (w2vShown !== key) { w2vShown = key; w2vSimilar(); w2vAnalogy(); w2vMap(); }
  }

  async function w2vSimilar() {
    const w = $('#w2v-word').value.trim(), box = $('#w2v-sim');
    if (!w) return;
    try {
      const r = await api('/api/w2v/similar?' + new URLSearchParams({ word: w, n: 15 }));
      box.innerHTML = `<p class="muted"><code>${esc(r.word)}</code> occurs ${fmt(r.count)} times in the training corpus.</p>
        <table class="grid num sim"><thead><tr><th>Word</th><th>Cosine similarity</th><th></th><th>Count</th><th></th></tr></thead><tbody>` +
        r.neighbours.map(n => `<tr><td><a href="#" data-w="${esc(n.word)}">${esc(n.word)}</a></td><td>${f2(n.similarity)}</td><td class="barcell"><i style="width:${Math.max(1, n.similarity * 100)}%"></i></td><td>${fmt(n.count)}</td><td><a href="#" data-s="${esc(n.word)}">search</a></td></tr>`).join('') + '</tbody></table>';
      $$('a[data-w]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#w2v-word').value = a.dataset.w; w2vSimilar(); }));
      $$('a[data-s]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); showTab('search'); doSearch(`"${a.dataset.s}"`); }));
    } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }
  async function w2vAnalogy() {
    const box = $('#w2v-ana');
    try {
      const r = await api('/api/w2v/analogy?' + new URLSearchParams({ a: $('#w2v-a').value.trim(), b: $('#w2v-b').value.trim(), c: $('#w2v-c').value.trim() }));
      box.innerHTML = `<table class="grid num sim"><thead><tr><th>${esc(r.a)} : ${esc(r.b)} = ${esc(r.c)} : ?</th><th>Cosine similarity</th><th></th></tr></thead><tbody>` +
        r.results.map(n => `<tr><td>${esc(n.word)}</td><td>${f2(n.similarity)}</td><td class="barcell"><i style="width:${Math.max(1, n.similarity * 100)}%"></i></td></tr>`).join('') + '</tbody></table>';
    } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }
  async function w2vMap() {
    const box = $('#w2v-map'), seeds = $('#w2v-seeds').value.trim().split(/[\s,;]+/).filter(Boolean).slice(0, 4);
    try {
      const r = await api('/api/w2v/projection?' + new URLSearchParams({ words: seeds.join(' '), n: 150 }));
      const groups = r.groups.length ? r.groups : [''];
      const series = groups.map((g, i) => ({ name: g ? `${g} + nearest neighbours` : '', color: SERIES[i], mode: 'dots', r: 4, opacity: 0.9,
        pts: r.points.filter(p => (p.group || '') === g).map(p => [p.x, p.y, p.word, !!p.seed, p.count]) }));
      plot(box, { title: 'word2vec embedding map', series, height: 460, width: 900, hover: 'nearest', labels: true, xlabel: 'Principal component 1', ylabel: 'Principal component 2',
        tipRows: (p, se) => [[se.color, p[2], ''], [null, fmt(p[4]), 'occurrences']] });
    } catch (e) { box.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }
  $('#w2v-train').addEventListener('click', async () => {
    const body = { model: $('#w2v-model').value, dim: +$('#w2v-dim').value, window: +$('#w2v-window').value, negative: +$('#w2v-negative').value, min_count: +$('#w2v-min').value, epochs: +$('#w2v-epochs').value };
    try { await api('/api/w2v/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); }
    catch (e) { alert(e.message); }
    w2vStatus();
  });
  $('#w2v-sim-btn').addEventListener('click', w2vSimilar);
  $('#w2v-word').addEventListener('keydown', e => { if (e.key === 'Enter') w2vSimilar(); });
  $('#w2v-ana-btn').addEventListener('click', w2vAnalogy);
  $('#w2v-map-btn').addEventListener('click', w2vMap);
  $('#w2v-seeds').addEventListener('keydown', e => { if (e.key === 'Enter') w2vMap(); });

  /* ================================================================== *
   * edit distance / spelling tab
   * ================================================================== */
  const OP_LABEL = { match: '相同', substitute: '取代', insert: '插入', delete: '刪除', transpose: '對調' };
  async function runEdit() {
    const box = $('#ed-out');
    try {
      const r = await api('/api/edit?' + new URLSearchParams({ a: $('#ed-a').value, b: $('#ed-b').value, transpose: $('#ed-tr').checked ? 1 : 0 }));
      const onPath = new Set(r.path.map(p => p.join(',')));
      let t = '<table class="dp"><tr><th></th><th>ε</th>' + [...r.b].map(ch => `<th>${esc(ch)}</th>`).join('') + '</tr>';
      r.matrix.forEach((row, i) => {
        t += `<tr><th>${i ? esc(r.a[i - 1]) : 'ε'}</th>` + row.map((v, j) => `<td class="${onPath.has(i + ',' + j) ? 'path' : ''}${i === r.a.length && j === r.b.length ? ' final' : ''}">${v}</td>`).join('') + '</tr>';
      });
      t += '</table>';
      const ops = r.ops.map(o => `<span class="op op-${o.op}" title="${OP_LABEL[o.op]}"><b>${esc(o.a || '∅')}</b><i>${o.op === 'match' ? '=' : '→'}</i><b>${esc(o.b || '∅')}</b><small>${OP_LABEL[o.op]}</small></span>`).join('');
      const n = k => r.ops.filter(o => o.op === k).length;
      box.innerHTML = `<div class="tiles"><div class="tile accent"><b>${r.distance}</b><span>edit distance(${esc(r.a)}, ${esc(r.b)})</span></div>
        <div class="tile"><b>${n('substitute')}</b><span>substitutions 取代</span></div><div class="tile"><b>${n('insert')}</b><span>insertions 插入</span></div>
        <div class="tile"><b>${n('delete')}</b><span>deletions 刪除</span></div><div class="tile"><b>${n('transpose')}</b><span>transpositions 對調</span></div></div>
        <h4>Alignment（一條最佳編輯路徑）</h4><div class="ops">${ops}</div><h4>DP table　<small class="muted">rows = a, columns = b</small></h4><div class="scroll">${t}</div>`;
    } catch (e) { box.textContent = e.message; }
  }
  async function runSpell() {
    const box = $('#sp-out'), w = $('#sp-word').value.trim();
    if (!w) return;
    try {
      const r = await api('/api/spell?' + new URLSearchParams({ word: w, k: $('#sp-k').value }));
      let h = `<p>${r.in_vocabulary ? `<code>${esc(r.word)}</code> <b>is in the dictionary</b> (collection frequency ${fmt(r.cf)}).` : `<code>${esc(r.word)}</code> is <b>not</b> in the dictionary.`}
        Dictionary: ${fmt(r.vocabulary)} words · the bigram index kept <b>${fmt(r.candidates)}</b> candidates for the DP step.</p>`;
      h += r.suggestions.length ? `<table class="grid num sim"><thead><tr><th>Suggestion</th><th>Edit distance</th><th>Collection frequency</th><th></th></tr></thead><tbody>` +
        r.suggestions.map((s, i) => `<tr class="${i ? '' : 'sel'}"><td><a href="#" data-e="${esc(s.word)}" title="show the DP table">${esc(s.word)}</a></td><td>${s.distance}</td><td>${fmt(s.cf)}</td><td><a href="#" data-s="${esc(s.word)}">search</a></td></tr>`).join('') + '</tbody></table>'
        : '<p class="muted">No dictionary word within the distance limit.</p>';
      box.innerHTML = h;
      $$('a[data-e]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#ed-a').value = r.word; $('#ed-b').value = a.dataset.e; runEdit(); window.scrollTo(0, 0); }));
      $$('a[data-s]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); showTab('search'); doSearch(a.dataset.s); }));
    } catch (e) { box.textContent = e.message; }
  }
  $('#ed-btn').addEventListener('click', runEdit);
  ['ed-a', 'ed-b'].forEach(id => $('#' + id).addEventListener('keydown', e => { if (e.key === 'Enter') runEdit(); }));
  $('#ed-tr').addEventListener('change', runEdit);
  $('#sp-btn').addEventListener('click', runSpell);
  $('#sp-word').addEventListener('keydown', e => { if (e.key === 'Enter') runSpell(); });
  $('#sp-k').addEventListener('change', runSpell);

  /* ---------------- build a PubMed collection ---------------- */
  $('#col-btn').addEventListener('click', async () => {
    const btn = $('#col-btn'), msg = $('#col-msg'), term = $('#col-term').value.trim();
    if (!term) { msg.textContent = 'Enter a PubMed query first.'; return; }
    btn.disabled = true; msg.textContent = 'Searching PubMed and downloading abstracts…';
    try {
      const r = await api('/api/collect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ term, n: +$('#col-n').value, name: $('#col-name').value.trim() }) });
      msg.textContent = `${fmt(r.fetched)} abstracts saved to data/${r.file} (PubMed has ${fmt(r.pubmed_matches)} matches) – index now holds ${fmt(r.documents)} documents. Retrain word2vec to use the new collection.`;
      state.corpus = null; zs.data = null;
      $('#indexed-refresh').click();
    } catch (e) { msg.textContent = e.message; }
    finally { btn.disabled = false; }
  });

  /* ---------------- tab activation ---------------- */
  let spellInit = false;
  document.addEventListener('tab', e => {
    if (e.detail === 'zipf') loadZipf();
    if (e.detail === 'w2v') w2vStatus();
    if (e.detail === 'spell' && !spellInit) { spellInit = true; runEdit(); runSpell(); }
  });
})();
