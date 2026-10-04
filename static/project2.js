/* Project #2 tabs: Zipf analysis, word2vec, edit distance / spelling correction */
(() => {
  const { $, $$, esc, fmt, api, state, showTab, doSearch } = window.IR;
  const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
  const f2 = (v, d = 3) => Number(v).toFixed(d);
  const pct = v => (v * 100).toFixed(1) + '%';
  const SVGNS = 'http://www.w3.org/2000/svg';

  /* ------------------------------------------------------------------ *
   * x/y plot (inline SVG): lines or dots, linear or log axes, hover tooltip
   * cfg = { series: [{ name, color, pts: [[x, y, label?]], mode: 'line'|'dots'|'fit', r }],
   *         xlog, ylog, xlabel, ylabel, height, vlines: [{ x, label }], hover: 'x'|'nearest',
   *         labels (draw point labels), tipX(x) -> text, tipY(y) -> text }
   * ------------------------------------------------------------------ */
  function niceTicks(a, b, n = 5) {
    const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw)), norm = raw / mag;
    const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  const short = v => { const a = Math.abs(v); return a >= 1e6 ? v / 1e6 + 'M' : a >= 1e3 ? v / 1e3 + 'k' : String(+v.toPrecision(3)); };

  function plot(el, cfg) {
    const W = cfg.width || 640, H = cfg.height || 340, P = { l: 58, r: 18, t: 16, b: 44 };
    const tx = v => (cfg.xlog ? Math.log10(v) : v), ty = v => (cfg.ylog ? Math.log10(v) : v);
    const all = cfg.series.flatMap(s => s.pts);
    if (!all.length) { el.innerHTML = '<p class="muted">no data</p>'; return; }
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    all.forEach(p => { const x = tx(p[0]), y = ty(p[1]); if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; });
    if (!cfg.xlog && cfg.xzero) x0 = Math.min(0, x0);
    if (!cfg.ylog && cfg.yzero) y0 = Math.min(0, y0);
    const px = (x1 - x0 || 1) * 0.03, py = (y1 - y0 || 1) * 0.05;
    if (cfg.xlog || !cfg.xzero) x0 -= px; x1 += px;
    if (cfg.ylog || !cfg.yzero) y0 -= py; y1 += py;
    const sx = v => P.l + (tx(v) - x0) / (x1 - x0) * (W - P.l - P.r);
    const sy = v => H - P.b - (ty(v) - y0) / (y1 - y0) * (H - P.t - P.b);
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
    s += `<text x="${(P.l + W - P.r) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="var(--muted)">${esc(cfg.xlabel || '')}</text>`;
    s += `<text transform="translate(14 ${(P.t + H - P.b) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="var(--muted)">${esc(cfg.ylabel || '')}</text>`;
    (cfg.vlines || []).forEach(v => {
      s += `<line x1="${sx(v.x)}" x2="${sx(v.x)}" y1="${P.t}" y2="${H - P.b}" stroke="var(--muted)" stroke-dasharray="3 4"/>`;
    });
    (cfg.bands || []).forEach(b => {
      s += `<text x="${(sx(b.from) + sx(b.to)) / 2}" y="${P.t + 12}" font-size="11" text-anchor="middle" fill="var(--muted)">${esc(b.label)}</text>`;
    });
    cfg.series.forEach(se => {
      if (se.mode === 'dots') {
        s += se.pts.map(p => `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${se.r || 3}" fill="${se.color}" fill-opacity="${se.opacity || 0.75}"/>`).join('');
      } else {
        const d = se.pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join('');
        s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.mode === 'fit' ? 1.5 : 2}" ${se.mode === 'fit' ? 'stroke-dasharray="6 4"' : ''} stroke-linejoin="round"/>`;
      }
    });
    if (cfg.labels) {                                    // point labels, skipping the ones that would collide
      const boxes = [];
      cfg.series.flatMap(se => se.pts).sort((p, q) => (q[3] ? 1 : 0) - (p[3] ? 1 : 0)).forEach(p => {      // emphasised labels first
        if (!p[2]) return;
        const x = sx(p[0]) + 6, y = sy(p[1]) + 4, w = String(p[2]).length * 6.2, box = [x, y - 10, x + w, y + 2];
        if (x + w > W - 2 || boxes.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) return;
        boxes.push(box);
        s += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="11" fill="var(--ink)" ${p[3] ? 'font-weight="700"' : ''}>${esc(p[2])}</text>`;
      });
    }
    s += `<line class="xh" y1="${P.t}" y2="${H - P.b}" stroke="var(--muted)" visibility="hidden"/><circle class="ring" r="6" fill="none" stroke="var(--ink)" stroke-width="2" visibility="hidden"/>`;
    s += `<rect class="hit" x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}" fill="transparent"/></svg>`;
    const named = cfg.series.filter(se => se.name);
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
        cfg.series.forEach(se => se.pts.forEach(p => { const d = (sx(p[0]) - mx) ** 2 + (sy(p[1]) - my) ** 2; if (d < bd) { bd = d; best = p; bs = se; } }));
        if (!best) return hide();
        ring.setAttribute('cx', sx(best[0])); ring.setAttribute('cy', sy(best[1])); ring.setAttribute('visibility', 'visible');
        (cfg.tipRows ? cfg.tipRows(best, bs) : [[bs.color, String(best[2] || ''), ''], [null, tipY(best[1]), tipX(best[0])]]).forEach(a => row(...a));
      } else {
        let anchor = null;
        cfg.series.forEach(se => {
          if (se.tip === false) return;
          let best = se.pts[0], bd = Infinity;
          se.pts.forEach(p => { const d = Math.abs(sx(p[0]) - mx); if (d < bd) { bd = d; best = p; } });
          if (!anchor) { anchor = best; const h = document.createElement('div'); h.className = 'tip-h'; h.textContent = tipX(best[0]) + (best[2] ? ' · ' + best[2] : ''); tip.appendChild(h); }
          row(se.color, tipY(best[1]), se.name || '');
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
  const zs = { data: null, cond: 'B', scale: 'logy', terms: '', table: null };
  const SEG_ZH = { high: '高頻', middle: '中頻', low: '低頻' };

  async function loadZipf(force) {
    if (zs.data && !force) return;
    $('#zipf').innerHTML = '<div class="card">Computing the Zipf analysis…</div>';
    try {
      zs.data = await api('/api/zipf');
      renderZipf();
      loadTerms();
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
      <div class="row"><b>Pre-processing condition</b><span class="segs">${condBtns}</span><span class="muted">${esc(c.description)}</span></div>
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
    </div>`;

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
    h += `<div class="card"><h2>Regression report　<small class="muted">log₁₀(CF) = a − b · log₁₀(r)</small></h2>
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

    h += `<div class="card"><h2>Top 50 terms by collection frequency　<small class="muted">condition ${c.key}</small></h2>
      <div class="scroll"><table class="grid num"><thead><tr><th>Rank r</th><th>Term</th><th>CF</th><th>DF</th><th>CF / DF</th><th>% of tokens</th><th>r × CF</th></tr></thead><tbody>
      ${c.top.map(([t, cf, df], i) => `<tr><td>${i + 1}</td><td><code>${esc(t)}</code></td><td>${fmt(cf)}</td><td>${fmt(df)}</td><td>${f2(cf / df, 2)}</td><td>${pct(cf / c.tokens)}</td><td>${fmt((i + 1) * cf)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="hint" style="margin-top:8px">若 k = 1，<code>r × CF</code> 應近似常數。</p></div>`;

    h += `<div class="card"><h2>Effect of pre-processing — conditions A / B / C / D</h2>
      <table class="grid num"><thead><tr><th>Condition</th><th>Tokens</th><th>Vocabulary</th><th>Avg tokens / doc</th><th>Hapax share</th><th>Top-10 share</th><th>Zipf exponent k</th><th>R²</th><th>RMSE</th><th>k (middle)</th><th>Top 8 terms</th></tr></thead><tbody>
      ${z.conditions.map((x, i) => `<tr class="${x.key === zs.cond ? 'sel' : ''}"><td><i class="key" style="background:${SERIES[i]}"></i><b>${x.key}</b> ${esc(x.name)}</td><td>${fmt(x.tokens)}</td><td>${fmt(x.vocabulary)}</td><td>${fmt(x.avg_tokens)}</td><td>${pct(x.hapax_share)}</td><td>${pct(x.top10_share)}</td><td><b>${f2(x.fit.exponent)}</b></td><td>${f2(x.fit.r2)}</td><td>${f2(x.fit.rmse)}</td><td>${f2(x.segments[1].exponent)}</td><td class="t">${x.top.slice(0, 8).map(t => `<code>${esc(t[0])}</code>`).join(' ')}</td></tr>`).join('')}
      </tbody></table>
      <div class="two" style="margin-top:14px"><div id="zp-overlay"></div>
      <div class="qa">
        <div><b>Vocabulary size</b>　A→B：去除標點後，<code>data,</code>、<code>data.</code>、<code>(data</code> 合併為同一個詞，詞彙由 ${fmt(byKey.A.vocabulary)} 降到 ${fmt(byKey.B.vocabulary)}（−${pct(1 - byKey.B.vocabulary / byKey.A.vocabulary)}）；token 數反而增加，因為連字號與斜線把 <code>glp-1</code> 切成兩個 token。
          B→C：停用詞只有 ${fmt(byKey.B.vocabulary - byKey.C.vocabulary)} 個詞型，詞彙幾乎不變，但 token 少了 ${pct(1 - byKey.C.tokens / byKey.B.tokens)}。C→D：Porter stemming 把詞形變化合併，詞彙再降到 ${fmt(byKey.D.vocabulary)}（−${pct(1 - byKey.D.vocabulary / byKey.C.vocabulary)}）。</div>
        <div><b>High-frequency terms</b>　A、B 的榜首是 the / of / and 等功能詞（前 10 名佔全部 token 的 ${pct(byKey.B.top10_share)}）；移除停用詞後，榜首換成主題詞（glp、receptor、weight、obesity…），前 10 名只佔 ${pct(byKey.C.top10_share)}；stemming 後同一詞幹的變形頻率相加（patient + patients → patient），高頻詞的 CF 進一步升高。</div>
        <div><b>Zipf exponent</b>　整體 k：${z.conditions.map(x => `${x.key} = ${f2(x.fit.exponent, 2)}`).join('、')}。A 最接近 1，是因為黏著標點的詞製造了大量只出現一次的詞型，拉長了尾巴；去標點後尾巴變短、斜率變陡。移除停用詞砍掉曲線最高的頭部，頭段更平（高頻段 k：B = ${f2(byKey.B.segments[0].exponent, 2)} → C = ${f2(byKey.C.segments[0].exponent, 2)}）。Stemming 縮短尾巴並抬高中段，整體 k 上升。</div>
        <div><b>Shape</b>　四條曲線的中段近乎平行；差異集中在頭部（停用詞移除後變平、出現「肩膀」）與尾部（詞彙越小，曲線越早觸底）。前處理改變的是截距與頭尾，不改變「少數詞極常見、多數詞極罕見」的長尾本質。</div>
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

    $('#zipf').innerHTML = h;
    $$('#zipf [data-cond]').forEach(b => b.addEventListener('click', () => { zs.cond = b.dataset.cond; renderZipf(); loadTerms(); }));
    $$('#zipf [data-scale]').forEach(b => b.addEventListener('click', () => { zs.scale = b.dataset.scale; renderZipf(); renderTerms(); }));
    $('#zp-terms-btn').addEventListener('click', () => { zs.terms = $('#zp-terms').value; loadTerms(); });
    $('#zp-terms').addEventListener('keydown', e => { if (e.key === 'Enter') { zs.terms = $('#zp-terms').value; loadTerms(); } });

    const idx = z.conditions.indexOf(c), color = SERIES[idx];
    const pts = c.points.map(([r, f]) => [r, f, r <= c.top.length ? c.top[r - 1][0] : '']);
    plot($('#zp-rank'), { title: 'Rank vs frequency', series: [{ color, pts, mode: 'line' }], ylog: zs.scale === 'logy', xzero: true, yzero: true,
      xlabel: 'Rank r', ylabel: 'Collection frequency CF', tipX: v => 'rank ' + fmt(v), tipY: v => 'CF ' + fmt(v) });
    plot($('#zp-loglog'), { title: 'Log-log rank-frequency plot', xlog: true, ylog: true, xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)',
      series: [{ name: 'Observed CF', color, pts, mode: 'line' }, { name: `Regression line (k = ${f2(c.fit.exponent)})`, color: 'var(--ink)', pts: fitLine(c), mode: 'fit', tip: false }],
      vlines: c.segments.slice(1).map(sg => ({ x: sg.from })), bands: c.segments.map(sg => ({ from: sg.from, to: sg.to, label: sg.name })),
      tipX: v => 'rank ' + fmt(v), tipY: v => 'CF ' + fmt(Math.round(v)) });
    plot($('#zp-overlay'), { title: 'Rank-frequency curves of the four conditions', xlog: true, ylog: true, xlabel: 'Rank r (log scale)', ylabel: 'CF (log scale)', height: 360,
      series: z.conditions.map((x, i) => ({ name: `${x.key} · ${x.name}`, color: SERIES[i], pts: x.points, mode: 'line' })),
      tipX: v => 'near rank ' + fmt(v), tipY: v => 'CF ' + fmt(v) });
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
