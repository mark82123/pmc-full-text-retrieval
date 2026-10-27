/* TF-IDF tab (Project #3): term statistics, term weighting of one document, the vector-space
   search engine, controlled experiments (one factor at a time), evaluation (P@k / R@k / MRR /
   MAP / nDCG) with label-based relevance judgements, and the research-question summary. */
(() => {
  const { $, $$, esc, fmt, api, showTab, plot, hbarChart } = window.IR;
  const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', '#0ea5e9', '#f43f5e', '#84cc16', '#a16207', '#64748b', '#be185d', '#0f766e'];
  const f = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const pct = v => (v == null ? '–' : (v * 100).toFixed(1) + '%');
  const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body == null ? undefined : JSON.stringify(body) });
  const err = (box, e) => { box.innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; };
  const st = { info: null, queries: null, evalData: null, loaded: false, lastSearch: null };
  const MEASURES = ['P@5', 'P@10', 'R@10', 'R@20', 'MRR', 'MAP', 'nDCG@10', 'nDCG@20', 'R-prec'];
  const colorOf = scheme => SERIES[Math.max(0, (st.info ? st.info.schemes.findIndex(s => s.scheme === scheme) : 0)) % SERIES.length];
  const gradeTag = (g, matched) => g == null ? '' : `<span class="grade g${g}" title="${esc((matched || []).join(', ') || 'not relevant')}">${g === 2 ? 'rel 2' : g === 1 ? 'rel 1' : 'not rel'}</span>`;
  const schemeLabel = s => s.scheme + ' · ' + s.name;

  /* ------------------------------------------------------------------ *
   * 1. collection, term statistics, tf / idf curves
   * ------------------------------------------------------------------ */
  async function loadInfo() {
    const box = $('#vsm-info');
    try {
      const i = await api('/api/vsm/info');
      st.info = i;
      fillSchemes();
      let h = `<div class="tiles kv-tiles">
        <div class="tile"><b>${fmt(i.documents)}</b><span>Documents (title + abstract) · ${esc(i.collection)}</span></div>
        <div class="tile"><b>${fmt(i.tokens)}</b><span>Tokens · ${fmt(i.stopword_tokens)} stop-word tokens removed (${pct(i.stopword_tokens / i.tokens)})</span></div>
        <div class="tile"><b>${fmt(i.index_terms)}</b><span>Index terms (after stemming) · ${i.avg_length} per document (${i.min_length}–${i.max_length})</span></div>
        <div class="tile accent"><b>${fmt(i.vocabulary)}</b><span>Vocabulary (distinct stems) · ${fmt(i.hapax_df)} with df = 1</span></div>
        <div class="tile"><b>${fmt(i.labelled)}</b><span>Documents with topic labels · MeSH ${fmt(i.with_mesh)}, keywords ${fmt(i.with_keywords)}</span></div>
        <div class="tile"><b>${i.build_time_s} s</b><span>Build time (pure Python)</span></div>
      </div>
      <div class="two">
        <div><h4>RQ1 — tf → weight：raw tf 線性成長，1 + ln tf 將 tf = 30 壓到 4.4（tf = 1 的 4.4 倍而非 30 倍）</h4><div id="vi-tf"></div></div>
        <div><h4>RQ2 — df → idf = ln(N / df)：出現在 ${fmt(i.documents)} 篇中 2,333 篇的 glp 幾乎沒有權重，df = 1 的詞 idf = ${f(Math.log(i.documents), 2)}</h4><div id="vi-idf"></div></div>
      </div>
      <div class="two" style="margin-top:10px">
        <div><h4>df 分布（詞彙表）</h4><table class="grid num"><tr><th>df</th><th>terms</th><th>share</th><th>idf range</th></tr>
          ${i.df_histogram.map(r => `<tr><td>${esc(r.bucket)}</td><td>${fmt(r.terms)}</td><td>${pct(r.terms / i.vocabulary)}</td><td>${f(r.idf_from, 2)} – ${f(r.idf_to, 2)}</td></tr>`).join('')}</table></div>
        <div><h4>df 最高的 25 個詞項（idf 最低）</h4><div class="chips">${i.top_df.map(r => `<span class="chip" title="cf ${fmt(r.cf)}, max tf ${r.max_tf}">${esc(r.term)} <small>df ${fmt(r.df)} · idf ${f(r.idf, 2)}</small></span>`).join('')}</div></div>
      </div>`;
      box.innerHTML = h;
      plot($('#vi-tf'), { title: 'tf transforms', height: 250, width: 600, xlabel: 'tf (occurrences in the document)', ylabel: 'tf weight', xzero: true, yzero: true,
        series: [
          { name: 'n  raw tf', color: SERIES[0], pts: i.tf_curve.map(c => [c.tf, c.raw]) },
          { name: 'l  1 + ln tf', color: SERIES[1], pts: i.tf_curve.map(c => [c.tf, c.log]) },
          { name: 'a  0.5 + 0.5 tf / max tf (max tf = 10)', color: SERIES[2], pts: i.tf_curve.map(c => [c.tf, c.aug_max10]) },
          { name: 'b  boolean', color: SERIES[3], pts: i.tf_curve.map(c => [c.tf, c.boolean]) }],
        tipX: v => `tf = ${v}`, tipY: v => f(v, 2) });
      plot($('#vi-idf'), { title: 'idf', height: 250, width: 600, xlabel: 'df (documents containing the term, log scale)', ylabel: 'idf', xlog: true, yzero: true,
        series: [
          { name: 't  idf = ln(N / df)', color: SERIES[0], pts: i.idf_curve.map(c => [c.df, c.idf]) },
          { name: 'p  prob idf = max(0, ln((N − df) / df))', color: SERIES[1], pts: i.idf_curve.map(c => [c.df, c.prob_idf]) }],
        tipX: v => `df = ${fmt(v)}`, tipY: v => f(v, 2) });
    } catch (e) { err(box, e); }
  }

  async function termStats() {
    const box = $('#vt-out');
    try {
      const r = await api('/api/vsm/terms?' + new URLSearchParams({ terms: $('#vt-terms').value }));
      const N = r.documents, maxIdf = Math.log(N);
      box.innerHTML = `<div class="tscroll"><table class="grid num"><tr><th>word</th><th>term (stem)</th><th>cf</th><th>df</th><th>df / N</th><th>idf = ln(N/df)</th><th>prob idf</th><th>max tf</th><th>1 + ln(max tf)</th><th>mean tf</th><th></th></tr>
        ${r.rows.map(t => t.stopword ? `<tr><td>${esc(t.word)}</td><td class="muted" colspan="10">stop word – removed before indexing</td></tr>`
          : `<tr><td>${esc(t.word)}</td><td>${esc(t.term)}${t.df ? '' : ' <span class="muted">(not in collection)</span>'}</td><td>${fmt(t.cf)}</td><td>${fmt(t.df)}</td><td>${pct(t.df_share)}</td><td>${f(t.idf, 3)}</td><td>${f(t.prob_idf, 3)}</td><td>${t.max_tf}</td><td>${f(t.log_max_tf, 2)}</td><td>${f(t.mean_tf, 2)}</td><td><span class="wbar" style="width:${Math.round(t.idf / maxIdf * 90)}px" title="idf"></span></td></tr>`).join('')}</table></div>
        <p class="hint" style="margin-top:6px">collection = ${fmt(N)} documents; the bar is idf relative to its maximum ln N = ${f(maxIdf, 2)} (a term in a single document).</p>`;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * scheme pickers
   * ------------------------------------------------------------------ */
  function fillSchemes() {
    const vec = st.info.schemes.filter(s => s.scheme !== 'bm25');
    $('#vq-scheme').innerHTML = st.info.schemes.map(s => `<option value="${s.scheme}"${s.scheme === st.info.default_scheme ? ' selected' : ''}>${esc(s.scheme)} — ${esc(s.name)}</option>`).join('');
    const chips = (id, list, on) => {
      const box = $('#' + id);
      box.innerHTML = list.map(s => `<label class="${on.includes(s.scheme) ? 'on' : ''}" title="${esc(s.formula)}"><input type="checkbox" value="${s.scheme}"${on.includes(s.scheme) ? ' checked' : ''}><i class="dot" style="background:${colorOf(s.scheme)};width:10px;height:10px;border-radius:50%;display:inline-block"></i>${esc(s.scheme)} <small class="muted">${esc(s.name)}</small></label>`).join('');
      $$('input', box).forEach(i => i.addEventListener('change', () => i.parentElement.classList.toggle('on', i.checked)));
    };
    chips('vd-schemes', vec, ['nnn', 'lnn', 'ntn', 'ltn', 'ltc']);
    chips('vc-schemes', st.info.schemes, ['nnn', 'lnn', 'ltn', 'ltc']);
    chips('ve-schemes', st.info.schemes, st.info.schemes.map(s => s.scheme));
  }
  const picked = id => $$('#' + id + ' input:checked').map(i => i.value);
  const setPicked = (id, list) => $$('#' + id + ' input').forEach(i => { i.checked = list.includes(i.value); i.parentElement.classList.toggle('on', i.checked); });

  /* ------------------------------------------------------------------ *
   * 2. one document under several schemes
   * ------------------------------------------------------------------ */
  async function docVectors() {
    const box = $('#vd-out');
    box.innerHTML = '<p class="muted">computing…</p>';
    try {
      const d = await api('/api/vsm/doc?' + new URLSearchParams({ id: $('#vd-id').value.trim(), schemes: picked('vd-schemes').join(','), top: $('#vd-top').value }));
      $('#vd-id').value = d.doc_id;
      let h = `<div class="note-box"><b>${esc(d.doc_id)}</b> · ${esc(d.title)} <span class="muted">(${esc(d.journal)} ${esc(d.year)})</span><br>
        <span class="muted">${fmt(d.tokens)} tokens → ${fmt(d.length)} index terms, ${fmt(d.unique_terms)} distinct, max tf = ${d.max_tf}</span>
        ${d.mesh.length ? `<div class="chips" style="margin-top:4px">${d.mesh.map(m => `<span class="chip${m.major ? ' term' : ''}" title="${m.major ? 'MeSH major topic' : 'MeSH'}">${esc(m.d)}${m.major ? '*' : ''}</span>`).join('')}</div>` : ''}
        ${d.keywords.length ? `<div class="chips" style="margin-top:4px">${d.keywords.map(k => `<span class="chip" title="author keyword">${esc(k)}</span>`).join('')}</div>` : ''}</div>`;
      h += '<div class="vcols">' + d.vectors.map((v, i) => {
        const max = v.vector[0] ? v.vector[0].weight : 1;
        return `<div class="vcol"><h4><span style="color:${colorOf(v.scheme)}">${esc(v.scheme)}</span> · ${esc(st.info.schemes.find(s => s.scheme === v.scheme)?.name || '')}</h4>
          <div class="muted">|d| = ${f(v.norm, 3)} · top term ${pct(v.top_share)} of the weight · top 5 ${pct(v.top5_share)}</div>
          <table class="grid num vtab"><tr><th>term</th><th>tf</th><th>df</th><th>w</th><th></th></tr>
          ${v.vector.map(r => `<tr><td title="${esc(r.term)}">${esc(r.term)}</td><td>${r.tf}</td><td>${fmt(r.df)}</td><td title="tf_w ${r.tf_w} × df_w ${r.df_w}">${f(r.weight, v.scheme.endsWith('c') ? 4 : 2)}</td><td class="bar"><span class="wbar" style="width:${Math.max(3, Math.round(r.weight / max * 100))}%;background:${colorOf(v.scheme)}"></span></td></tr>`).join('')}</table></div>`;
      }).join('') + '</div>';
      const vec = d.vectors;
      if (vec.length >= 2) {
        h += `<p class="hint" style="margin-top:8px">同一篇文件、同樣的詞，只改 weighting：${vec.map(v => `<b>${esc(v.scheme)}</b> 首詞「${esc(v.vector[0]?.term || '')}」佔 ${pct(v.top_share)}`).join('；')}。</p>`;
      }
      box.innerHTML = h;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 3. search
   * ------------------------------------------------------------------ */
  async function search() {
    const box = $('#vq-out');
    const q = $('#vq').value.trim();
    if (!q) return;
    box.innerHTML = '<p class="muted">searching…</p>';
    try {
      const r = await api('/api/vsm/search?' + new URLSearchParams({ q, scheme: $('#vq-scheme').value, qscheme: $('#vq-qscheme').value, k: $('#vq-k').value }));
      st.lastSearch = r;
      const m = r.measures;
      let h = `<div class="analysis">${r.terms.map(t => t.term ? `<span class="chip term">${esc(t.word)} → <b>${esc(t.term)}</b> <small>df ${fmt(t.df)} · idf ${f(t.idf, 2)}${t.role === 'unknown' ? ' · not in collection' : ''}</small></span>` : `<span class="chip stopword">${esc(t.word)}</span>`).join('')}
        <span class="muted">query vector (${esc(r.qscheme)}): ${Object.entries(r.qvec).map(([t, w]) => `${esc(t)} ${f(w, 3)}`).join(', ') || '–'}</span></div>
        <div class="summary">${fmt(r.total)} documents with a non-zero score in ${r.time_ms} ms · document scheme <b>${esc(r.scheme)}</b> = ${esc(r.scheme_info.name)} (tf: ${esc(r.scheme_info.tf)}; df: ${esc(r.scheme_info.df)}; norm: ${esc(r.scheme_info.norm)})
        ${m ? `<br>evaluation query <b>${esc(r.query_id)}</b>: ${fmt(m.relevant)} relevant documents · P@5 ${f(m['P@5'], 2)} · P@10 ${f(m['P@10'], 2)} · R@10 ${f(m['R@10'], 3)} · RR ${f(m.RR, 2)} · AP ${f(m.AP, 3)} · nDCG@10 ${f(m['nDCG@10'], 3)}` : ''}</div>`;
      const top = r.results[0] ? r.results[0].score : 1;
      h += r.results.map(d => {
        const total = d.contributions.reduce((a, c) => a + c.part, 0) || 1;
        return `<div class="vsm-res" data-doc="${esc(d.doc_id)}">
          <div class="rank">#${d.rank} · score <b>${f(d.score, 4)}</b> ${gradeTag(d.grade, d.matched_labels)}
            ${r.query_id ? `<span class="judge" title="manual relevance judgement for ${esc(r.query_id)}">${[0, 1, 2].map(g => `<button data-g="${g}" class="${d.grade === g && d.matched_labels && d.matched_labels[0] === 'manual' ? 'on' : ''}">${g}</button>`).join('')}<button data-g="" title="remove the manual judgement">×</button></span>` : ''}</div>
          <div class="scorebar"><i style="width:${Math.round(d.score / top * 100)}%"></i></div>
          <h3>${esc(d.title)}</h3>
          <div class="meta"><span>${esc(d.journal)} (${esc(d.year)})</span><span>${esc(d.doc_id)}</span><span>${fmt(d.length)} index terms · max tf ${d.max_tf}</span><span>${d.matched} of ${r.terms.filter(t => t.term).length} query terms</span></div>
          <div class="cbar">${d.contributions.map((c, i) => `<i style="width:${(c.part / total * 100).toFixed(1)}%;background:${SERIES[i % SERIES.length]}" title="${esc(c.term)}: ${f(c.part, 4)}"></i>`).join('')}</div>
          <div class="contrib">${d.contributions.map((c, i) => `<span class="chip"><i class="dot" style="background:${SERIES[i % SERIES.length]};width:8px;height:8px;border-radius:50%;display:inline-block"></i>${esc(c.term)} <small>tf ${c.tf} · df ${fmt(c.df)}</small> ${f(c.qw, 3)} × ${f(c.dw, 3)} = <b>${f(c.part, 4)}</b></span>`).join('')}</div>
          <details><summary>abstract · MeSH · keywords</summary><div class="abs">${d.abstract.map(p => `<p>${esc(p)}</p>`).join('')}</div>
            ${d.mesh.length ? `<div class="chips">${d.mesh.map(x => `<span class="chip${x.endsWith('*') ? ' term' : ''}">${esc(x)}</span>`).join('')}</div>` : '<p class="muted">no MeSH headings (not yet indexed)</p>'}
            ${d.keywords.length ? `<div class="chips" style="margin-top:4px">${d.keywords.map(x => `<span class="chip">${esc(x)}</span>`).join('')}</div>` : ''}</details>
        </div>`;
      }).join('');
      box.innerHTML = h;
      $$('.judge button', box).forEach(b => b.addEventListener('click', async () => {
        const doc = b.closest('.vsm-res').dataset.doc;
        try { await post('/api/vsm/judge?' + new URLSearchParams({ id: r.query_id, doc, grade: b.dataset.g })); st.evalData = null; st.queries = null; search(); loadQueries(); }
        catch (e) { alert(e.message); }
      }));
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 4. controlled experiments
   * ------------------------------------------------------------------ */
  async function compare() {
    const box = $('#vc-out');
    const q = $('#vc-q').value.trim(), schemes = picked('vc-schemes');
    if (!q || !schemes.length) return;
    box.innerHTML = '<p class="muted">ranking…</p>';
    try {
      const r = await api('/api/vsm/compare?' + new URLSearchParams({ q, schemes: schemes.join(','), k: $('#vc-k').value }));
      const base = r.schemes[0];
      let h = `<div class="analysis">${r.terms.map(t => t.term ? `<span class="chip term">${esc(t.word)} → <b>${esc(t.term)}</b> <small>df ${fmt(t.df)} · idf ${f(t.idf, 2)}</small></span>` : `<span class="chip stopword">${esc(t.word)}</span>`).join('')}
        <span class="muted">base = ${esc(r.base)}; mean document length ${r.avg_length}${r.query_id ? `; evaluation query ${esc(r.query_id)}` : ''}</span></div>
        <div class="tscroll"><table class="grid num"><tr><th>scheme</th><th>name</th><th>scored docs</th><th>overlap@${r.k} vs ${esc(r.base)}</th><th>Kendall τ</th><th>mean length of top ${r.k}</th><th>mean max tf</th>${r.query_id ? '<th>P@5</th><th>P@10</th><th>R@10</th><th>AP</th><th>nDCG@10</th>' : ''}</tr>
        ${r.schemes.map(s => `<tr><td><b style="color:${colorOf(s.scheme)}">${esc(s.scheme)}</b></td><td>${esc(s.info.name)}</td><td>${fmt(s.total)}</td><td>${f(s.overlap, 2)}</td><td>${f(s.tau, 2)}</td><td>${s.mean_length}</td><td>${s.mean_max_tf}</td>
          ${s.measures ? `<td>${f(s.measures['P@5'], 2)}</td><td>${f(s.measures['P@10'], 2)}</td><td>${f(s.measures['R@10'], 3)}</td><td>${f(s.measures.AP, 3)}</td><td>${f(s.measures['nDCG@10'], 3)}</td>` : ''}</tr>`).join('')}</table></div>`;
      h += '<div class="vcols">' + r.schemes.map(s => `<div class="vcol"><h4><span style="color:${colorOf(s.scheme)}">${esc(s.scheme)}</span> · ${esc(s.info.name)}</h4>
        <ol>${s.top.map(t => `<li class="${t.grade ? 'rel' + t.grade : ''}" title="${esc(t.title)}">${esc(t.title.length > 70 ? t.title.slice(0, 70) + '…' : t.title)}
          <span class="mv">score ${f(t.score, 3)} · len ${t.length} · max tf ${t.max_tf}${s.scheme !== r.base ? ` · ${t.rank_base ? '#' + t.rank_base + ' under ' + esc(r.base) : 'not in ' + esc(r.base)}` : ''}${t.grade != null ? ' · rel ' + t.grade : ''}</span></li>`).join('')}</ol></div>`).join('') + '</div>';
      h += `<p class="hint" style="margin-top:8px">綠色 = 相關文件（深綠 grade 2）。mean length：${r.schemes.map(s => `${esc(s.scheme)} ${s.mean_length}`).join(' · ')}（全集合平均 ${r.avg_length}）。</p>`;
      box.innerHTML = h;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 5. evaluation
   * ------------------------------------------------------------------ */
  async function loadQueries() {
    const box = $('#ve-queries');
    try {
      const r = await api('/api/vsm/queries');
      st.queries = r;
      $('#vq-examples').innerHTML = '<span class="muted">evaluation queries:</span> ' + r.queries.map(q => `<a href="#" data-q="${esc(q.text)}" title="${esc(q.description)} · ${q.relevant} relevant">${esc(q.id)}</a>`).join('');
      $$('#vq-examples a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#vq').value = a.dataset.q; search(); }));
      box.innerHTML = `<div class="tscroll"><table class="grid qtab"><tr><th>id</th><th>query</th><th style="text-align:right">relevant</th><th style="text-align:right">grade 2 / 1</th><th style="text-align:right">via MeSH</th><th style="text-align:right">keywords only</th><th style="text-align:right">manual</th><th>rule (every group must match a label)</th><th></th></tr>
        ${r.queries.map(q => `<tr><td>${esc(q.id)}</td><td class="t"><b>${esc(q.text)}</b><br><span class="muted">${esc(q.description)}</span></td><td style="text-align:right">${q.relevant}</td><td style="text-align:right">${q.grade2} / ${q.grade1}</td><td style="text-align:right">${q.from_mesh}</td><td style="text-align:right">${q.from_keyword_only}</td><td style="text-align:right">${q.manual}</td>
          <td class="t rule">${q.groups.map(g => '(' + g.map(esc).join(' | ') + ')').join(' AND ')}</td>
          <td><a href="#" data-rel="${esc(q.id)}">relevant set</a> · <a href="#" data-run="${esc(q.text)}">search</a></td></tr>`).join('')}</table></div>
        <p class="hint" style="margin-top:6px">${fmt(r.labelled)} of ${fmt(r.documents)} documents carry at least one label (${pct(r.labelled / r.documents)}); the rest can never be judged relevant. File: <code>${esc(r.file)}</code></p>`;
      $$('a[data-rel]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); relevantSet(a.dataset.rel); }));
      $$('a[data-run]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#vq').value = a.dataset.run; search(); location.hash = '#vsm-s3'; $('#vsm-s3').scrollIntoView(); }));
    } catch (e) { err(box, e); }
  }

  async function relevantSet(qid) {
    const box = $('#ve-rel');
    box.innerHTML = '<p class="muted">loading…</p>';
    try {
      const scheme = $('#vq-scheme').value;
      const r = await api('/api/vsm/relevant?' + new URLSearchParams({ id: qid, scheme }));
      const m = r.measures;
      box.innerHTML = `<h4 style="margin:12px 0 4px">Relevant set of ${esc(r.query.id)} — “${esc(r.query.text)}” (${m.relevant} documents; rank under <b>${esc(scheme)}</b>: P@10 ${f(m['P@10'], 2)}, R@10 ${f(m['R@10'], 3)}, AP ${f(m.AP, 3)}, nDCG@10 ${f(m['nDCG@10'], 3)}; ${fmt(r.retrieved)} documents scored)</h4>
        <div class="tscroll"><table class="grid"><tr><th style="text-align:right">rank</th><th>grade</th><th>document</th><th style="text-align:right">length</th><th>matched labels (* = MeSH major topic)</th></tr>
        ${r.relevant.map(d => `<tr><td style="text-align:right">${d.rank || '<span class="muted">not retrieved</span>'}</td><td>${gradeTag(d.grade, d.matched)}${d.manual ? ' <small class="muted">manual</small>' : ''}</td><td class="t">${esc(d.title)} <span class="muted">${esc(d.doc_id)}</span></td><td style="text-align:right">${d.length}</td><td class="t" style="font-size:12px">${d.matched.map(esc).join(', ')}</td></tr>`).join('')}</table></div>`;
      box.scrollIntoView({ block: 'start' });
    } catch (e) { err(box, e); }
  }

  async function evaluate() {
    const box = $('#ve-out');
    const schemes = picked('ve-schemes');
    if (!schemes.length) return;
    box.innerHTML = '<p class="muted">evaluating…</p>';
    try {
      const r = await api('/api/vsm/eval?' + new URLSearchParams({ schemes: schemes.join(','), k: $('#ve-k').value, condensed: $('#ve-cond').checked ? 1 : 0 }));
      st.evalData = r;
      const best = {};
      MEASURES.forEach(k => { best[k] = Math.max(...r.schemes.map(s => s.mean[k])); });
      let h = `<div class="tscroll"><table class="grid num" id="ve-table"><tr><th>scheme</th><th>name</th>${MEASURES.map(k => `<th>${k}</th>`).join('')}</tr>
        ${r.schemes.map(s => `<tr><td><b style="color:${colorOf(s.scheme)}">${esc(s.scheme)}</b></td><td>${esc(s.info.name)}</td>${MEASURES.map(k => `<td class="${s.mean[k] === best[k] ? 'best' : ''}">${f(s.mean[k], 3)}</td>`).join('')}</tr>`).join('')}</table></div>
        <p class="hint">mean over ${r.queries.length} queries${r.condensed ? ' (condensed lists: unlabelled documents removed before measuring)' : ' (unlabelled documents count as not relevant)'}; best value per column in bold.</p>
        <div class="row" style="margin:8px 0"><label>Chart / per-query measure <select id="ve-measure">${MEASURES.map(k => `<option${k === 'nDCG@10' ? ' selected' : ''}>${k}</option>`).join('')}</select></label></div>
        <div class="two"><div id="ve-chart"></div><div id="ve-chart2"></div></div>
        <div id="ve-perq"></div>`;
      box.innerHTML = h;
      const draw = () => {
        const k = $('#ve-measure').value;
        $('#ve-chart').innerHTML = `<h4>${esc(k)} per scheme</h4>` + hbarChart(r.schemes.map(s => [s.scheme + ' ' + s.info.name, +s.mean[k].toFixed(3)]), { width: 560, barH: 16, labelW: 190 });
        // factor view: tf (n/l) × df (n/t) × norm (n/c)
        const grid = ['nnn', 'lnn', 'ntn', 'ltn', 'nnc', 'lnc', 'ntc', 'ltc'].filter(s => r.schemes.some(x => x.scheme === s));
        const val = s => r.schemes.find(x => x.scheme === s)?.mean[k];
        $('#ve-chart2').innerHTML = grid.length >= 4 ? `<h4>${esc(k)} — 2 × 2 × 2 factor grid（tf × idf × norm）</h4>
          <table class="grid num"><tr><th>tf \\ idf · norm</th><th>no idf, no norm</th><th>idf, no norm</th><th>no idf, L2</th><th>idf, L2</th></tr>
          ${[['n', 'raw tf'], ['l', 'log tf']].map(([t, name]) => `<tr><td>${name}</td>${[t + 'nn', t + 'tn', t + 'nc', t + 'tc'].map(s => `<td>${val(s) != null ? f(val(s), 3) : '–'}</td>`).join('')}</tr>`).join('')}</table>
          <p class="hint">讀法：同一列比較 idf / 正規化的效果，同一欄比較 raw vs log tf。</p>` : '';
        $('#ve-perq').innerHTML = `<h4>${esc(k)} per query</h4><div class="tscroll"><table class="grid num"><tr><th>query</th>${r.schemes.map(s => `<th style="color:${colorOf(s.scheme)}">${esc(s.scheme)}</th>`).join('')}<th>relevant</th></tr>
          ${r.queries.map((q, i) => { const vals = r.schemes.map(s => s.per_query[i][k]); const mx = Math.max(...vals);
            return `<tr><td title="${esc(q.text)}">${esc(q.id)} <span class="muted">${esc(q.text.length > 44 ? q.text.slice(0, 44) + '…' : q.text)}</span></td>${vals.map(v => `<td class="${v === mx ? 'best' : ''}">${f(v, 3)}</td>`).join('')}<td>${q.relevant}</td></tr>`; }).join('')}</table></div>`;
      };
      $('#ve-measure').addEventListener('change', draw);
      draw();
      researchQuestions();
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 6. research questions from the live numbers
   * ------------------------------------------------------------------ */
  function researchQuestions() {
    const box = $('#vr-out');
    const r = st.evalData, i = st.info;
    if (!r || !i) return;
    const M = s => r.schemes.find(x => x.scheme === s)?.mean;
    const has = (...s) => s.every(x => M(x));
    const cmp = (a, b, k) => has(a, b) ? `${k} ${f(M(a)[k], 3)} → ${f(M(b)[k], 3)} (${M(b)[k] >= M(a)[k] ? '+' : ''}${((M(b)[k] - M(a)[k]) / (M(a)[k] || 1) * 100).toFixed(0)}%)` : '–';
    const tf30 = i.tf_curve.find(c => c.tf === 30);
    const rank = [...r.schemes].sort((a, b) => b.mean['nDCG@10'] - a.mean['nDCG@10']);
    const rankMap = [...r.schemes].sort((a, b) => b.mean.MAP - a.mean.MAP);
    const glp = i.top_df.find(r => !/^\d+$/.test(r.term)) || i.top_df[0];
    box.innerHTML = `<div class="rq">
      <p><b>RQ1 原始 TF 與 Log TF。</b> raw tf 讓出現 30 次的詞權重是出現 1 次的 30 倍；1 + ln tf 只給 <span class="n">${f(tf30?.log, 2)}</span> 倍，因為詞的重要性不隨次數等比例增加（第 2 節的文件向量可見首詞佔比的變化）。
        在檢索上：${cmp('nnn', 'lnn', 'MAP')}、${cmp('nnn', 'lnn', 'nDCG@10')}；加上 idf 後 ${cmp('ntn', 'ltn', 'MAP')}；再加 L2 ${cmp('ntc', 'ltc', 'MAP')}。</p>
      <p><b>RQ2 IDF。</b> idf = ln(N / df) 隨 df 單調下降：<span class="n">${esc(glp?.term)}</span> 出現在 ${fmt(glp?.df)} / ${fmt(i.documents)} 篇，idf = ${f(glp?.idf, 2)}，幾乎不影響排名；df = 1 的詞 idf = ${f(Math.log(i.documents), 2)}。
        沒有 idf 時，像 receptor、agonist、diabetes 這類在本集合裡出現於半數以上文件的詞和真正有辨識力的詞（tirzepatide、apnea）權重相同，排名由文件長度與常見詞的 tf 決定。
        檢索效果：${cmp('lnn', 'ltn', 'MAP')}、${cmp('lnn', 'ltn', 'nDCG@10')}；正規化後 ${cmp('lnc', 'ltc', 'MAP')}。</p>
      <p><b>RQ3 TF、IDF 或 TF-IDF。</b> MAP：raw TF (nnn) ${f(M('nnn')?.MAP, 3)}、log TF (lnn) ${f(M('lnn')?.MAP, 3)}、IDF only (btn) ${f(M('btn')?.MAP, 3)}、TF-IDF (ltn) ${f(M('ltn')?.MAP, 3)}、TF-IDF + L2 (ltc) ${f(M('ltc')?.MAP, 3)}；
        nDCG@10：nnn ${f(M('nnn')?.['nDCG@10'], 3)}、lnn ${f(M('lnn')?.['nDCG@10'], 3)}、btn ${f(M('btn')?.['nDCG@10'], 3)}、ltn ${f(M('ltn')?.['nDCG@10'], 3)}、ltc ${f(M('ltc')?.['nDCG@10'], 3)}。
        MAP 最佳：<b>${esc(rankMap[0]?.scheme)}</b>（${esc(rankMap[0]?.info.name)}）；nDCG@10 最佳：<b>${esc(rank[0]?.scheme)}</b>。</p>
      <p><b>RQ4 L2 正規化。</b> 未正規化的內積 = |d|·cos θ，隨文件長度增加（第 4 節：同一查詢下 ltn 的前 10 名平均長度明顯高於 ltc，兩者前 10 名重疊不到一半）；正規化後長短文件只比方向，但極短的文件（30 個詞的更正啟事）會因此竄到第一名。
        指標：${cmp('ltn', 'ltc', 'MAP')}、${cmp('ltn', 'ltc', 'P@10')}、${cmp('nnn', 'nnc', 'MAP')}。本集合全是摘要、長度相近（平均 ${i.avg_length}），長度偏差本來就小，正規化的收益被短文件竄升抵銷；在混入全文的集合裡（report/rq4_mixed_lengths.py）未正規化時長文佔據前 10 名，正規化後為 0。BM25 的部分正規化（b = 0.75）兩種情況都好。</p>
      <p><b>RQ5 各指標。</b> 以 ${r.queries.length} 個查詢的平均：${MEASURES.map(k => `${k} 最佳 <b>${esc([...r.schemes].sort((a, b) => b.mean[k] - a.mean[k])[0]?.scheme)}</b>`).join('、')}。
        不同指標看重不同位置：P@k / nDCG@k 只看前 k 名，MRR 只看第一個相關文件，MAP 看整個排名；結論須結合查詢、資料集與指標判斷——第 5 節的每題結果顯示各 scheme 的優劣並非在每個查詢上都一致。${has('bm25') ? ` 參考：BM25 MAP ${f(M('bm25').MAP, 3)}、nDCG@10 ${f(M('bm25')['nDCG@10'], 3)}。` : ''}</p>
    </div>`;
  }

  /* ------------------------------------------------------------------ *
   * wiring
   * ------------------------------------------------------------------ */
  $('#vt-go').addEventListener('click', termStats);
  $('#vt-terms').addEventListener('keydown', e => { if (e.key === 'Enter') termStats(); });
  $('#vd-go').addEventListener('click', docVectors);
  $('#vd-id').addEventListener('keydown', e => { if (e.key === 'Enter') docVectors(); });
  $('#vq-go').addEventListener('click', search);
  $('#vq').addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
  $('#vq-scheme').addEventListener('change', () => { if (st.lastSearch) search(); });
  $('#vq-qscheme').addEventListener('change', () => { if (st.lastSearch) search(); });
  $('#vc-go').addEventListener('click', compare);
  $('#vc-q').addEventListener('keydown', e => { if (e.key === 'Enter') compare(); });
  $$('#vsm-s4 button[data-exp]').forEach(b => b.addEventListener('click', () => { setPicked('vc-schemes', b.dataset.exp.split(',')); compare(); }));
  $('#ve-go').addEventListener('click', evaluate);

  async function init() {
    if (st.loaded) return;
    st.loaded = true;
    await loadInfo();
    await Promise.all([termStats(), docVectors(), loadQueries()]);
    await search();
    await compare();
    await evaluate();
  }
  document.addEventListener('tab', e => { if (e.detail === 'vsm') init(); });
  if (location.hash === '#vsm' || location.hash.startsWith('#vsm-')) { showTab('vsm'); }
})();
