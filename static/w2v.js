/* Word-embedding tab (Project #2, word2vec): pre-processing + vocabulary, training examples,
   model training queue + registry, CBOW vs skip-gram, neighbours side by side, cosine matrix,
   PCA / t-SNE maps, frequency-bucket evaluation and neighbour overlap. */
(() => {
  const { $, $$, esc, fmt, api, showTab, doSearch, plot } = window.IR;
  const SERIES = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', '#0ea5e9', '#f43f5e', '#84cc16', '#a16207'];
  const f = (v, d = 3) => (v == null || Number.isNaN(v) ? '–' : Number(v).toFixed(d));
  const pct = v => (v == null ? '–' : (v * 100).toFixed(1) + '%');
  const BASE_SG = 'sg-d50-w5-m5-e10', BASE_CBOW = 'cbow-d50-w5-m5-e10';
  const st = { models: [], active: null, timer: null, shownKey: '', vocab: null, examples: null, centre: null, evalCache: {} };
  const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body == null ? undefined : JSON.stringify(body) });
  const err = (box, e) => { box.innerHTML = `<p class="muted">${esc(e.message || e)}</p>`; };
  const colorOf = name => SERIES[Math.max(0, st.models.findIndex(m => m.name === name)) % SERIES.length];
  const modelLabel = m => `${m.model === 'skip-gram' ? 'Skip-gram' : 'CBOW'} · ${m.dim}d · win ${m.window} · min ${m.min_count}${m.seed && m.seed !== 1 ? ' · seed ' + m.seed : ''}${m.stopwords_removed === false ? ' · stop words kept' : ''}`;

  /* ------------------------------------------------------------------ *
   * model selectors shared by several sections
   * ------------------------------------------------------------------ */
  function fillSelects() {
    const names = st.models.map(m => m.name);
    $$('select.model-select').forEach(sel => {
      const prev = sel.value;
      const want = prev && names.includes(prev) ? prev
        : sel.id === 'ov-a' && names.includes(BASE_SG) ? BASE_SG
        : sel.id === 'ov-b' && names.includes(BASE_CBOW) ? BASE_CBOW
        : names.includes(st.active) ? st.active : names[0] || '';
      sel.innerHTML = names.map(n => `<option value="${esc(n)}"${n === want ? ' selected' : ''}>${esc(n)}</option>`).join('');
      sel.disabled = !names.length;
    });
    ['nn-models', 'ev-models'].forEach(id => {
      const box = $('#' + id), picked = new Set($$('input:checked', box).map(i => i.value));
      const first = !box.children.length;
      box.innerHTML = st.models.map(m => {
        const on = first ? (m.name === BASE_SG || m.name === BASE_CBOW || (!names.includes(BASE_SG) && m.name === st.active)) : picked.has(m.name);
        return `<label class="${on ? 'on' : ''}" title="${esc(modelLabel(m))}"><input type="checkbox" value="${esc(m.name)}"${on ? ' checked' : ''}><i class="dot" style="background:${colorOf(m.name)};width:10px;height:10px;border-radius:50%;display:inline-block"></i>${esc(m.name)}</label>`;
      }).join('');
      $$('input', box).forEach(i => i.addEventListener('change', () => i.parentElement.classList.toggle('on', i.checked)));
    });
  }
  const picked = id => $$('#' + id + ' input:checked').map(i => i.value);

  /* ------------------------------------------------------------------ *
   * 1. pre-processing + vocabulary
   * ------------------------------------------------------------------ */
  async function loadVocab() {
    const box = $('#vb-out');
    try {
      const v = await api('/api/w2v/vocab?' + new URLSearchParams({ min_count: $('#vb-min').value, stopwords: $('#vb-stop').checked ? 1 : 0 }));
      st.vocab = v;
      const sw = v.stopword_tokens != null;
      let h = `<div class="tiles kv-tiles">
        <div class="tile"><b>${fmt(v.documents)}</b><span>Documents (title + abstract)</span></div>
        <div class="tile"><b>${fmt(v.sentences)}</b><span>Sentences · ${v.avg_sentence_length} tokens each</span></div>
        <div class="tile"><b>${fmt(v.tokens)}</b><span>Tokens${sw ? ` (after removing ${fmt(v.stopword_tokens)} stop-word tokens = ${pct(v.stopword_tokens / v.tokens_with_stopwords)})` : ''}</span></div>
        <div class="tile"><b>${fmt(v.types)}</b><span>Distinct words (types) · hapax ${pct(v.hapax_share)}</span></div>
        <div class="tile accent"><b>${fmt(v.vocabulary)}</b><span>Vocabulary at min_count ${v.min_count} · covers ${pct(v.coverage)} of tokens</span></div>
        <div class="tile"><b>${fmt(v.examples[1].skipgram_dynamic)}</b><span>Skip-gram pairs / epoch (window 5, dynamic)</span></div>
        <div class="tile"><b>${fmt(v.examples[1].cbow)}</b><span>CBOW examples / epoch (= kept tokens)</span></div>
      </div>
      <div class="two">
        <div><h4>Minimum count: what stays in the vocabulary</h4>
          <table class="grid num"><thead><tr><th>min_count</th><th>types kept</th><th>share of types</th><th>tokens covered</th><th>share of tokens</th></tr></thead><tbody>` +
        v.thresholds.map(t => `<tr${t.min_count === v.min_count ? ' class="sel"' : ''}><td>≥ ${t.min_count}</td><td>${fmt(t.types)}</td><td>${pct(t.types_share)}</td><td>${fmt(t.tokens)}</td><td>${pct(t.token_share)}</td></tr>`).join('') +
        `</tbody></table>
          <p class="hint" style="margin-top:8px">Zipf 的長尾：${pct(v.hapax_share)} 的詞型只出現一次。min_count 5 丟掉約 ${pct(1 - v.thresholds[3].types_share)} 的詞型，但只損失 ${pct(1 - v.thresholds[3].token_share)} 的 token——這些詞出現太少，向量學不穩，也拖慢 softmax / 負樣本表。</p>
          <h4>Training examples per epoch</h4>
          <table class="grid num"><thead><tr><th>window</th><th>skip-gram pairs (fixed ±w)</th><th>skip-gram pairs (dynamic, expected)</th><th>CBOW examples</th></tr></thead><tbody>` +
        v.examples.map(e => `<tr><td>±${e.window}</td><td>${fmt(e.skipgram_fixed)}</td><td>${fmt(e.skipgram_dynamic)}</td><td>${fmt(e.cbow)}</td></tr>`).join('') +
        `</tbody></table></div>
        <div><h4>Most frequent words (${v.stopwords_removed ? 'stop words removed' : 'stop words kept'})</h4>${window.IR.hbarChart(v.top_words.slice(0, 30), { width: 520, barH: 14, labelW: 120 })}</div>
      </div>`;
      if (v.walk) {
        const w = v.walk, show = (arr, cls) => arr.map(t => `<span class="tok ${cls ? cls(t) : ''}">${esc(t)}</span>`).join('');
        h += `<h4>Pipeline on one document: <a href="#" data-doc="${esc(w.doc_id)}">${esc(w.doc_id)}</a> — ${esc(w.title)} <small class="muted">(${w.sentences} sentences; first ${w.steps.length} shown)</small></h4><div class="pipe">`;
        w.steps.forEach((s, i) => {
          const lowerSet = new Set(s.no_stopwords), keptSet = new Set(s.kept);
          h += `<div style="margin:6px 0 10px"><span class="lbl">sentence ${i + 1}</span>${esc(s.sentence)}<br>
            <span class="lbl">tokens (${s.tokens.length})</span>${show(s.tokens)}<br>
            <span class="lbl">lower-case</span>${show(s.lower, t => (v.stopwords_removed && !lowerSet.has(t) ? 'drop' : ''))}<br>
            <span class="lbl">kept (count ≥ ${v.min_count})</span>${show(v.stopwords_removed ? s.no_stopwords : s.lower.filter(t => t.length > 1), t => (keptSet.has(t) ? '' : 'drop rare'))}</div>`;
        });
        h += '</div>';
      }
      box.innerHTML = h;
      $$('a[data-doc]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); window.IR.state.prevTab = 'w2v'; showTab('search'); doSearch(`"${a.dataset.doc.replace(/^PMID/, '')}"`); }));
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 2. training examples
   * ------------------------------------------------------------------ */
  async function genExamples() {
    const box = $('#ex-out');
    try {
      const r = await api('/api/w2v/examples?' + new URLSearchParams({ text: $('#ex-text').value, window: $('#ex-win').value, stopwords: $('#ex-stop').checked ? 1 : 0 }));
      st.examples = r; st.centre = null;
      renderExamples();
    } catch (e) { err(box, e); }
  }
  function renderExamples() {
    const r = st.examples, box = $('#ex-out'), c = st.centre;
    const ctx = c == null ? new Set() : new Set(r.cbow.find(e => e.position === c)?.offsets.map(o => o + c) || []);
    const sent = r.words.map((w, i) => `<span class="w ${i === c ? 'centre' : ctx.has(i) ? 'ctx' : ''}" data-i="${i}">${esc(w)}</span>`).join('');
    const sg = r.skipgram.filter(e => c == null || e.position === c), cb = r.cbow.filter(e => c == null || e.position === c);
    box.innerHTML = `<p class="muted">After pre-processing: ${r.words.length} words${r.sentences > 1 ? ` (only the first of ${r.sentences} sentences is used)` : ''}. Fixed window ±${r.window} → <b>${r.n_skipgram}</b> skip-gram pairs, <b>${r.n_cbow}</b> CBOW examples${c != null ? ` — showing position ${c} (<a href="#" id="ex-all">show all</a>)` : ' — click a word to see its examples only'}.</p>
      <div class="ex-sent">${sent}</div>
      <div class="two">
        <div><h4>Skip-gram：input = 中心詞，target = 一個周圍詞（${sg.length} pairs）</h4><div class="scroll" style="max-height:340px"><table class="grid"><thead><tr><th>#</th><th>input (centre)</th><th>→ target (context)</th><th>offset</th></tr></thead><tbody>` +
      sg.map((e, i) => `<tr><td>${i + 1}</td><td>${esc(e.centre)}</td><td>${esc(e.context)}</td><td>${e.offset > 0 ? '+' : ''}${e.offset}</td></tr>`).join('') + `</tbody></table></div></div>
        <div><h4>CBOW：input = 周圍詞的平均向量，target = 中心詞（${cb.length} examples）</h4><div class="scroll" style="max-height:340px"><table class="grid"><thead><tr><th>#</th><th>input (context words)</th><th>→ target (centre)</th></tr></thead><tbody>` +
      cb.map((e, i) => `<tr><td>${i + 1}</td><td class="t">${e.context.map(esc).join(' , ')}</td><td>${esc(e.centre)}</td></tr>`).join('') + `</tbody></table></div></div>
      </div>
      <p class="hint" style="margin-top:8px">每個 Skip-gram 配對與每個 CBOW 樣本都還會配上 k 個負樣本（預設 5，從 unigram<sup>0.75</sup> 分布抽），所以一個句子的實際更新次數是表中數字 × (1 + k)。Skip-gram 的樣本數約是 CBOW 的 2w 倍——這就是它訓練較慢、但罕見詞被更新較多次的原因。</p>`;
    $$('.ex-sent .w', box).forEach(el => el.addEventListener('click', () => { st.centre = st.centre === +el.dataset.i ? null : +el.dataset.i; renderExamples(); }));
    const all = $('#ex-all'); if (all) all.addEventListener('click', e => { e.preventDefault(); st.centre = null; renderExamples(); });
  }

  /* ------------------------------------------------------------------ *
   * 3. training: queue, registry, loss curves
   * ------------------------------------------------------------------ */
  async function status() {
    let s;
    try { s = await api('/api/w2v/status'); } catch (e) { err($('#w2v-info'), e); return; }
    st.models = s.models; st.active = s.active;
    fillSelects();
    // jobs
    const jb = $('#w2v-jobs');
    const live = s.jobs.filter(j => j.state === 'queued' || j.state === 'running' || j.state === 'error' || (j.state === 'done' && j.elapsed != null));
    jb.innerHTML = live.length ? '<div class="jobs">' + live.slice(-8).map(j => `<div class="job"><b style="min-width:200px">${esc(j.name)}</b><span class="state-${j.state}" style="min-width:70px">${j.state}</span>
        <div class="bar"><i style="width:${(j.progress * 100).toFixed(1)}%"></i></div><span class="muted" style="min-width:300px">${j.state === 'running' ? `${(j.progress * 100).toFixed(1)}% · epoch ${j.epoch || 1}/${j.params.epochs}${j.loss != null ? ` · loss ${j.loss} · α ${j.alpha}` : ''} · ${j.elapsed}s` : j.state === 'error' ? esc(j.message) : j.state === 'done' ? `done in ${j.elapsed}s` : 'waiting'}</span></div>`).join('') + '</div>' : '';
    $('#w2v-train').disabled = false;
    clearTimeout(st.timer);
    if (s.jobs.some(j => j.state === 'running' || j.state === 'queued')) st.timer = setTimeout(status, 1500);
    // models table
    const info = $('#w2v-info');
    if (!s.models.length) { info.innerHTML = `<p class="muted">No model yet. Training corpus: ${esc(s.collection)} – ${fmt(s.documents)} documents. Press <b>Train</b> or <b>Queue comparison set</b>.</p>`; return; }
    info.innerHTML = `<div class="hscroll"><table class="grid num models"><thead><tr><th>active</th><th>name</th><th>architecture</th><th>dim</th><th>window</th><th>min count</th><th>neg</th><th>epochs</th><th>seed</th><th>stop words</th><th>vocabulary</th><th>examples / epoch</th><th>final loss</th><th>time</th><th>trained</th><th></th></tr></thead><tbody>` +
      s.models.map(m => `<tr${m.name === s.active ? ' class="sel"' : ''}><td class="act"><input type="radio" name="w2v-active" value="${esc(m.name)}"${m.name === s.active ? ' checked' : ''}></td>
        <td><i class="dot" style="background:${colorOf(m.name)};width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px"></i>${esc(m.name)}</td><td>${m.model}</td><td>${m.dim}</td><td>±${m.window}</td><td>${m.min_count}</td><td>${m.negative}</td><td>${m.epochs}</td><td>${m.seed || 1}</td><td>${m.stopwords_removed === false ? 'kept' : 'removed'}</td>
        <td>${fmt(m.vocabulary)}</td><td>${fmt(m.pairs_per_epoch)}</td><td>${f(m.final_loss, 4)}</td><td>${fmt(m.train_seconds)} s</td><td>${esc((m.trained || '').slice(0, 16))}</td><td><a href="#" data-del="${esc(m.name)}" title="delete this model">✕</a></td></tr>`).join('') +
      `</tbody></table></div>
      <div class="two" style="margin-top:12px"><div><h4>Training loss per epoch（mean negative-sampling loss per example）</h4><div id="w2v-loss"></div>
        <p class="hint">loss = −log σ(h·v<sub>target</sub>) − Σ<sub>k</sub> log σ(−h·v<sub>neg</sub>)。CBOW 的 h 是視窗平均，較平滑、較易預測，所以 loss 數值通常比 Skip-gram 低——兩者的 loss 不能直接比較「誰學得好」，只能看各自是否收斂。</p></div>
        <div><h4>Most frequent words in the active model (${esc(s.active)})</h4><div class="chips">${(s.model ? s.model.top_words : []).map(([w, n]) => `<span class="chip term"><a href="#" data-w="${esc(w)}">${esc(w)}</a> <small>${fmt(n)}</small></span>`).join('')}</div></div></div>`;
    plot($('#w2v-loss'), { title: 'Training loss', height: 260, width: 640, xlabel: 'Epoch', ylabel: 'Mean loss per example', xzero: false,
      series: s.models.map(m => ({ name: m.name, color: colorOf(m.name), mode: 'line', width: m.name === s.active ? 3 : 1.5, opacity: m.model === 'cbow' ? 0.9 : 1, pts: m.loss_history.map((l, i) => [i + 1, l]) })),
      tipX: v => 'epoch ' + v, tipY: v => f(v, 4) });
    $$('input[name=w2v-active]', info).forEach(r => r.addEventListener('change', async () => { await post('/api/w2v/select?name=' + encodeURIComponent(r.value)); status(); }));
    $$('a[data-del]', info).forEach(a => a.addEventListener('click', async e => { e.preventDefault(); if (!confirm(`Delete model ${a.dataset.del}?`)) return; await post('/api/w2v/delete?name=' + encodeURIComponent(a.dataset.del)); st.evalCache = {}; status(); }));
    $$('a[data-w]', info).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#w2v-word').value = a.dataset.w; similar(); }));
    const key = s.models.map(m => m.name + m.train_seconds).join('|') + '#' + s.active;
    if (st.shownKey !== key) { st.shownKey = key; comparisonTable(); similar(); analogy(); compare(); pairs(); maps(); evaluate(); overlap(); }
  }
  function sweepJobs() {
    const have = new Set(st.models.map(m => m.name));
    const want = [];
    for (const model of ['sg', 'cbow']) for (const window of [5, 2]) want.push({ model, dim: 50, window, min_count: 5, epochs: 10 });
    for (const model of ['sg', 'cbow']) want.push({ model, dim: 100, window: 5, min_count: 5, epochs: 10 });
    want.push({ model: 'sg', dim: 50, window: 5, min_count: 2, epochs: 10 });
    for (const model of ['sg', 'cbow']) want.push({ model, dim: 50, window: 5, min_count: 5, epochs: 10, seed: 2 });
    return want.filter(j => !have.has(`${j.model}-d${j.dim}-w${j.window}-m${j.min_count}-e${j.epochs}${j.seed ? '-s' + j.seed : ''}`));
  }
  $('#w2v-train').addEventListener('click', async () => {
    const body = { model: $('#w2v-model').value, dim: +$('#w2v-dim').value, window: +$('#w2v-window').value, negative: +$('#w2v-negative').value, min_count: +$('#w2v-min').value,
      epochs: +$('#w2v-epochs').value, seed: +$('#w2v-seed').value, remove_stopwords: $('#w2v-stop').checked, name: $('#w2v-name').value.trim() };
    try { await post('/api/w2v/train', body); } catch (e) { alert(e.message); }
    status();
  });
  $('#w2v-sweep').addEventListener('click', async () => {
    const jobs = sweepJobs();
    if (!jobs.length) return alert('The comparison set is already trained.');
    if (!confirm(`Queue ${jobs.length} training jobs (about ${Math.round(jobs.reduce((a, j) => a + (j.model === 'sg' ? 4.5 : 1.5) * (j.dim / 50) * (j.window === 2 ? 0.6 : 1), 0))} minutes in pure Python)?`)) return;
    try { await post('/api/w2v/train', jobs); } catch (e) { alert(e.message); }
    status();
  });

  /* ------------------------------------------------------------------ *
   * 4. CBOW vs skip-gram table (static properties + measured values)
   * ------------------------------------------------------------------ */
  function comparisonTable() {
    const rows = [
      ['Input', '視窗內 2b 個周圍詞的向量<b>平均</b>（bag of words，不分順序）', '一個<b>中心詞</b>的向量'],
      ['Prediction target', '中心詞', '視窗內的<b>每一個</b>周圍詞（各自一個配對）'],
      ['Training examples', '每個位置 1 個樣本 ≈ 語料 token 數（本語料 window 5：約 15.9 萬 / epoch）', '每個位置 2b 個配對 ≈ token 數 × 平均視窗（window 5 dynamic：約 80 萬 / epoch）'],
      ['Training speed', '快（樣本少；平均後只做一次 1 + k 的 dot product）', '慢，約為 CBOW 的 2b 倍（本實作 50 維 10 epochs：CBOW ≈ 1.5 分鐘，Skip-gram ≈ 4–5 分鐘）'],
      ['Rare words', '差：罕見詞只分到平均梯度的 1/2b，且被同視窗高頻詞稀釋；其向量更新少', '好：罕見詞每出現一次自身向量就更新 2b 次，各帶 k 個負樣本'],
      ['Expected behaviour', '高頻詞、句法 / 功能類的相似度較平滑；loss 較低；大語料下效果與 Skip-gram 相近', '語意 / 主題相似度與專有名詞（藥名、基因、縮寫）較佳；小語料較穩；對 window、subsampling 較敏感'],
    ];
    $('#cmp-table').innerHTML = `<table class="grid cmp"><thead><tr><th>Property</th><th>CBOW (Continuous Bag-of-Words)</th><th>Skip-gram</th></tr></thead><tbody>` +
      rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('') + '</tbody></table>';
    measured();
  }
  async function measured() {
    const box = $('#cmp-measured'), sg = st.models.find(m => m.name === BASE_SG), cb = st.models.find(m => m.name === BASE_CBOW);
    if (!sg || !cb) return;
    try {
      const ev = await evalFor([BASE_SG, BASE_CBOW], 10);
      const b = (m, label) => (m.buckets.find(x => x.bucket === label) || {});
      const [es, ec] = ev.models;
      const row = (name, a, c, fmtv = v => v) => `<tr><td>${name}</td><td>${fmtv(c)}</td><td>${fmtv(a)}</td></tr>`;
      box.innerHTML = `<table class="grid num"><thead><tr><th>measure</th><th>CBOW</th><th>Skip-gram</th></tr></thead><tbody>` +
        row('training examples / epoch', sg.pairs_per_epoch, cb.pairs_per_epoch, fmt) + row('training time (10 epochs)', sg.train_seconds + ' s', cb.train_seconds + ' s') +
        row('final loss', sg.final_loss, cb.final_loss, v => f(v, 4)) + row('vocabulary', sg.vocabulary, cb.vocabulary, fmt) +
        row('sim@10, words seen 5–9 times', b(es, '5-9').sim_at_k, b(ec, '5-9').sim_at_k, v => f(v)) +
        row('median neighbour count, words seen 5–9 times', b(es, '5-9').neighbour_count, b(ec, '5-9').neighbour_count, fmt) +
        row('variant hit@10, words seen 5–9 times', b(es, '5-9').variant_hit, b(ec, '5-9').variant_hit, pct) +
        row('variant hit@10, words seen 20–49 times', b(es, '20-49').variant_hit, b(ec, '20-49').variant_hit, pct) +
        row('variant hit@10, words seen 200+ times', b(es, '200+').variant_hit, b(ec, '200+').variant_hit, pct) +
        `</tbody></table><p class="hint" style="margin-top:6px">variant hit@10 = 同 Porter 詞幹的其他詞形出現在前 10 近鄰的比例（第 8 節有完整分桶表）。</p>`;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 5. nearest neighbours side by side; single-model neighbours; analogy
   * ------------------------------------------------------------------ */
  async function compare() {
    const box = $('#nn-out'), models = picked('nn-models');
    if (!models.length) { box.innerHTML = '<p class="muted">tick at least one model</p>'; return; }
    try {
      const r = await api('/api/w2v/compare?' + new URLSearchParams({ words: $('#nn-words').value, models: models.join(','), n: $('#nn-n').value }));
      const names = r.models.map(m => m.name);
      const shared = row => { const sets = names.map(n => row.models[n]).filter(Boolean).map(m => new Set(m.neighbours.map(x => x.word))); return sets.length < 2 ? new Set() : new Set([...sets[0]].filter(w => sets.every(s => s.has(w)))); };
      box.innerHTML = `<div class="hscroll"><table class="grid nn"><thead><tr><th>word</th><th>count</th>${names.map(n => `<th><i class="dot" style="background:${colorOf(n)};width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px"></i>${esc(n)}</th>`).join('')}${names.length > 1 ? '<th>overlap (Jaccard)</th>' : ''}</tr></thead><tbody>` +
        r.words.map(row => {
          const sh = shared(row), cnt = names.map(n => row.models[n]).find(Boolean);
          return `<tr><td><b>${esc(row.word)}</b></td><td>${cnt ? fmt(cnt.count) : '–'}</td>` + names.map(n => {
            const m = row.models[n];
            return m ? `<td>${m.neighbours.map(x => `<span class="nb ${sh.has(x.word) ? 'shared' : ''}"><a href="#" data-w="${esc(x.word)}">${esc(x.word)}</a> <small>${f(x.similarity, 2)}</small></span>`).join('')}</td>` : '<td class="oov">not in vocabulary</td>';
          }).join('') + (names.length > 1 ? `<td>${Object.entries(row.overlap).map(([k, o]) => `<div title="${esc(k)}">${f(o.jaccard, 2)}</div>`).join('') || '–'}</td>` : '') + '</tr>';
        }).join('') + '</tbody></table></div>';
      $$('a[data-w]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#w2v-word').value = a.dataset.w; similar(); $('#w2v-word').scrollIntoView({ block: 'center', behavior: 'smooth' }); }));
    } catch (e) { err(box, e); }
  }
  async function similar() {
    const w = $('#w2v-word').value.trim(), box = $('#w2v-sim');
    if (!w) return;
    try {
      const r = await api('/api/w2v/similar?' + new URLSearchParams({ word: w, n: 15, model: $('#sim-model').value }));
      box.innerHTML = `<p class="muted"><code>${esc(r.word)}</code> occurs ${fmt(r.count)} times in the training corpus · model ${esc(r.model)}.</p>
        <table class="grid num sim"><thead><tr><th>Word</th><th>Cosine similarity</th><th></th><th>Count</th><th></th></tr></thead><tbody>` +
        r.neighbours.map(n => `<tr><td><a href="#" data-w="${esc(n.word)}">${esc(n.word)}</a></td><td>${f(n.similarity)}</td><td class="barcell"><i style="width:${Math.max(1, n.similarity * 100)}%"></i></td><td>${fmt(n.count)}</td><td><a href="#" data-s="${esc(n.word)}">search</a></td></tr>`).join('') + '</tbody></table>';
      $$('a[data-w]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('#w2v-word').value = a.dataset.w; similar(); }));
      $$('a[data-s]', box).forEach(a => a.addEventListener('click', e => { e.preventDefault(); window.IR.state.prevTab = 'w2v'; showTab('search'); doSearch(`"${a.dataset.s}"`); }));
    } catch (e) { err(box, e); }
  }
  async function analogy() {
    const box = $('#w2v-ana');
    try {
      const r = await api('/api/w2v/analogy?' + new URLSearchParams({ a: $('#w2v-a').value.trim(), b: $('#w2v-b').value.trim(), c: $('#w2v-c').value.trim(), model: $('#ana-model').value }));
      box.innerHTML = `<table class="grid num sim"><thead><tr><th>${esc(r.a)} : ${esc(r.b)} = ${esc(r.c)} : ?</th><th>Cosine similarity</th><th></th></tr></thead><tbody>` +
        r.results.map(n => `<tr><td>${esc(n.word)}</td><td>${f(n.similarity)}</td><td class="barcell"><i style="width:${Math.max(1, n.similarity * 100)}%"></i></td></tr>`).join('') + '</tbody></table>';
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 6. cosine matrix of word groups
   * ------------------------------------------------------------------ */
  const heat = v => { // cosine -> background; dark blue = 1, white = 0, grey-ish for negative
    if (v < 0) return ['hsl(0 0% ' + (96 - Math.min(1, -v) * 30) + '%)', 'var(--ink)'];
    const l = 97 - v * 57; return [`hsl(215 75% ${l}%)`, l < 62 ? '#fff' : 'var(--ink)'];
  };
  async function pairs() {
    const box = $('#pr-out');
    try {
      const r = await api('/api/w2v/pairs?' + new URLSearchParams({ groups: $('#pr-groups').value, model: $('#pr-model').value }));
      const n = r.words.length;
      let h = `<div class="tiles kv-tiles"><div class="tile accent"><b>${f(r.mean_within)}</b><span>mean cosine within groups</span></div><div class="tile"><b>${f(r.mean_between)}</b><span>mean cosine between groups</span></div>
        <div class="tile"><b>${f(r.mean_random)}</b><span>mean cosine of 400 random vocabulary pairs</span></div><div class="tile"><b>${n}</b><span>words in ${r.groups.length} groups · model ${esc(r.model)}</span></div></div>`;
      h += '<div class="hscroll"><table class="heat"><thead><tr><th></th>' + r.words.map((w, j) => `<th class="ch ${j && r.group[j] !== r.group[j - 1] ? 'gsep' : ''}" style="color:${SERIES[r.group[j] % SERIES.length]}"><span>${esc(w)}</span></th>`).join('') + '</tr></thead><tbody>';
      r.matrix.forEach((row, i) => {
        h += `<tr class="${i && r.group[i] !== r.group[i - 1] ? 'gsep' : ''}"><th class="rh" style="color:${SERIES[r.group[i] % SERIES.length]}">${esc(r.words[i])} <small class="muted">${fmt(r.counts[i])}</small></th>` +
          row.map((v, j) => { const [bg, fg] = i === j ? ['var(--surface3)', 'var(--muted)'] : heat(v); return `<td class="${j && r.group[j] !== r.group[j - 1] ? 'gsep' : ''}" style="background:${bg};color:${fg}" title="${esc(r.words[i])} · ${esc(r.words[j])}: ${f(v)}">${i === j ? '' : f(v, 2)}</td>`; }).join('') + '</tr>';
      });
      h += '</tbody></table></div>';
      h += '<div class="two" style="margin-top:10px"><div><h4>Per group</h4><table class="grid num"><thead><tr><th>group</th><th>words</th><th>mean within</th></tr></thead><tbody>' +
        r.groups.map((g, i) => `<tr><td><i class="dot" style="background:${SERIES[i % SERIES.length]};width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px"></i>${i + 1}</td><td class="t">${g.words.map(esc).join(', ')}</td><td>${f(g.mean_within)}</td></tr>`).join('') + '</tbody></table>' +
        (r.missing.length ? `<p class="muted">not in vocabulary: ${r.missing.map(esc).join(', ')}</p>` : '') + '</div>' +
        '<div><h4>Highest cross-group pairs（模型認為相關、但你分在不同組的詞）</h4><table class="grid num"><thead><tr><th>a</th><th>b</th><th>cosine</th></tr></thead><tbody>' +
        r.pairs.filter(p => !p.same_group).slice(0, 8).map(p => `<tr><td>${esc(p.a)}</td><td>${esc(p.b)}</td><td>${f(p.similarity)}</td></tr>`).join('') + '</tbody></table>' +
        '<h4>Lowest within-group pairs（同組卻不相近）</h4><table class="grid num"><thead><tr><th>a</th><th>b</th><th>cosine</th></tr></thead><tbody>' +
        r.pairs.filter(p => p.same_group).slice(-5).reverse().map(p => `<tr><td>${esc(p.a)}</td><td>${esc(p.b)}</td><td>${f(p.similarity)}</td></tr>`).join('') + '</tbody></table></div></div>';
      box.innerHTML = h;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * 7. PCA and t-SNE maps
   * ------------------------------------------------------------------ */
  async function maps() {
    const seeds = $('#map-seeds').value.trim().split(/[\s,;]+/).filter(Boolean).slice(0, 6);
    const base = { words: seeds.join(' '), n: $('#map-n').value, model: $('#map-model').value, neighbours: $('#map-k').value, perplexity: $('#map-perp').value, iters: $('#map-iters').value, seed: $('#map-seed').value };
    $('#map-tsne').innerHTML = '<p class="muted">computing t-SNE…</p>';
    const draw = (el, r, title) => {
      const groups = r.groups.length ? r.groups : [''];
      const series = groups.map((g, i) => ({ name: g ? `${g} + neighbours` : 'most frequent words', color: SERIES[i], mode: 'dots', r: 4, opacity: 0.9,
        pts: r.points.filter(p => (p.group || '') === g).map(p => [p.x, p.y, p.word, !!p.seed, p.count, p.knn]) }));
      plot(el, { title, series, height: 480, width: 620, hover: 'nearest', labels: true, xlabel: r.method === 'pca' ? `PC1 (${pct(r.explained_variance[0])} of variance)` : 't-SNE 1', ylabel: r.method === 'pca' ? `PC2 (${pct(r.explained_variance[1])})` : 't-SNE 2',
        tipRows: (p, se) => [[se.color, p[2], ''], [null, fmt(p[4]), 'occurrences'], [null, pct(p[5]), 'of its 10 nearest neighbours kept in 2-D']] });
      el.insertAdjacentHTML('beforeend', `<div class="tiles kv-tiles"><div class="tile"><b>${pct(r.knn_preservation.k5)}</b><span>k-NN preservation, k = 5</span></div><div class="tile"><b>${pct(r.knn_preservation.k10)}</b><span>k = 10</span></div>` +
        (r.method === 'pca' ? `<div class="tile"><b>${pct(r.explained_variance[0] + r.explained_variance[1])}</b><span>variance explained by PC1 + PC2</span></div>` : `<div class="tile"><b>${f(r.kl.at(-1), 3)}</b><span>final KL(P‖Q) · perplexity ${r.perplexity} · ${r.iters} iterations · ${r.seconds}s</span></div>`) + '</div>');
    };
    try {
      const p = await api('/api/w2v/projection?' + new URLSearchParams({ ...base, method: 'pca' }));
      draw($('#map-pca'), p, `PCA — ${p.model}`);
      const t = await api('/api/w2v/projection?' + new URLSearchParams({ ...base, method: 'tsne' }));
      draw($('#map-tsne'), t, `t-SNE — ${t.model}`);
      $('#map-notes').innerHTML = `<div class="note-box" style="margin-top:10px"><b>怎麼讀這兩張圖。</b> PCA 的兩軸是向量空間中變異最大的兩個方向，距離有全域意義，但 ${p.points.length} 個 ${st.models.find(m => m.name === p.model)?.dim || ''} 維向量只有 ${pct(p.explained_variance[0] + p.explained_variance[1])} 的變異被保留，所以不同群常疊在一起、鄰域只保留 ${pct(p.knn_preservation.k10)}。
        t-SNE 把高維的鄰居機率 P 與低維的 Q 拉近（KL 散度 ${f(t.kl.at(-1), 3)}），鄰域保留 ${pct(t.knn_preservation.k10)}，群看起來分明——但群與群之間的距離、群的大小、以及沒有鄰居關係的點之間的相對位置都沒有意義，換 perplexity 或 seed 會得到不同的圖。
        視覺上成群≠語意好：t-SNE 對隨機向量也能畫出「群」，請搭配第 6 節的 cosine 矩陣與第 8 節的量化指標判斷。</div>`;
    } catch (e) { err($('#map-tsne'), e); if (!$('#map-pca').children.length) err($('#map-pca'), e); }
  }

  /* ------------------------------------------------------------------ *
   * 8. evaluation per frequency bucket; neighbour overlap between two models
   * ------------------------------------------------------------------ */
  async function evalFor(names, k) {
    const key = names.join(',') + '#' + k;
    if (!st.evalCache[key]) st.evalCache[key] = api('/api/w2v/eval?' + new URLSearchParams({ models: names.join(','), k }));
    try { return await st.evalCache[key]; } catch (e) { delete st.evalCache[key]; throw e; }
  }
  async function evaluate() {
    const box = $('#ev-out'), names = picked('ev-models');
    if (!names.length) { box.innerHTML = '<p class="muted">tick at least one model</p>'; return; }
    try {
      const r = await evalFor(names, +$('#ev-k').value);
      const buckets = [...new Set(r.models.flatMap(m => m.buckets.map(b => b.bucket)))];
      const metric = (label, get, fmtv, hint) => `<h4 title="${esc(hint)}">${label}</h4><table class="grid num"><thead><tr><th>model</th>${buckets.map(b => `<th>${b}</th>`).join('')}</tr></thead><tbody>` +
        r.models.map(m => `<tr><td><i class="dot" style="background:${colorOf(m.name)};width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px"></i>${esc(m.name)}</td>${buckets.map(b => { const x = m.buckets.find(y => y.bucket === b); return `<td>${x ? fmtv(get(x)) : '–'}</td>`; }).join('')}</tr>`).join('') + '</tbody></table>';
      const counts = r.models[0];
      box.innerHTML = `<p class="muted">Words per frequency bucket (${esc(counts.name)}): ${counts.buckets.map(b => `${b.bucket}: ${fmt(b.words)}`).join(' · ')}; ${fmt(counts.stem_groups)} stem groups cover ${fmt(counts.words_with_variant)} words. k = ${r.k}.</p>
        <div class="two">
          <div>${metric('sim@k — mean cosine of the k nearest neighbours', b => b.sim_at_k, v => f(v), 'how tight the neighbourhood is; very high values for rare words mean they cluster among themselves')}
          ${metric('median collection frequency of the neighbours', b => b.neighbour_count, fmt, 'do rare words get rare or frequent neighbours?')}</div>
          <div>${metric('variant hit@k — a same-stem word form is among the k neighbours', b => b.variant_hit, pct, 'share of words (with at least one variant in the vocabulary) whose variant is in the top k')}
          ${metric('variant MRR — 1 / rank of the first variant', b => b.variant_mrr, v => f(v), 'mean reciprocal rank, 0 when no variant is in the top k')}</div>
        </div>
        <h4>Examples of variants found</h4><div class="chips">` + r.models.map(m => m.buckets.flatMap(b => b.examples.slice(0, 2).map(e => `<span class="chip" title="${esc(m.name)} · bucket ${b.bucket}">${esc(e.word)} → ${esc(e.variant)} <small>#${e.rank} · ${f(e.similarity, 2)}</small></span>`)).join('')).join('') + '</div>';
      // bar chart: variant hit by bucket, one series per model
      const pts = r.models.map(m => ({ name: m.name, color: colorOf(m.name), mode: 'line', pts: buckets.map((b, i) => [i + 1, (m.buckets.find(y => y.bucket === b) || {}).variant_hit || 0]) }));
      const el = document.createElement('div'); box.appendChild(el);
      plot(el, { title: 'variant hit@k by frequency bucket', height: 240, width: 640, series: pts, xlabel: 'frequency bucket: ' + buckets.map((b, i) => `${i + 1} = ${b}`).join(', '), ylabel: 'variant hit@k', yzero: true, tipX: v => buckets[Math.round(v) - 1] || '', tipY: v => pct(v) });
    } catch (e) { err(box, e); }
  }
  async function overlap() {
    const box = $('#ov-out'), a = $('#ov-a').value, b = $('#ov-b').value;
    if (!a || !b) return;
    if (a === b) { box.innerHTML = '<p class="muted">choose two different models</p>'; return; }
    try {
      const r = await api('/api/w2v/overlap?' + new URLSearchParams({ models: a + ',' + b, k: $('#ev-k').value }));
      box.innerHTML = `<div class="tiles kv-tiles"><div class="tile accent"><b>${pct(r.overall)}</b><span>mean Jaccard of top-${r.k} neighbours · ${fmt(r.shared_vocabulary)} shared words (sampled 120 per bucket)</span></div>` +
        r.buckets.map(x => `<div class="tile"><b>${pct(x.overlap)}</b><span>count ${x.bucket} (${fmt(x.words)} words)</span></div>`).join('') + '</div>' +
        `<div class="two"><div><h4>Most stable words</h4><div class="chips">${r.most_stable.map(x => `<span class="chip term">${esc(x.word)} <small>${pct(x.overlap)} · ${fmt(x.count)}×</small></span>`).join('')}</div></div>
        <div><h4>Least stable words</h4><div class="chips">${r.least_stable.map(x => `<span class="chip">${esc(x.word)} <small>${pct(x.overlap)} · ${fmt(x.count)}×</small></span>`).join('')}</div></div></div>
        <p class="hint" style="margin-top:8px">同設定、不同 seed 的重疊率是「穩定度」的上限參考：若兩個 seed 對罕見詞的近鄰只有二到三成重疊，表示這些詞的向量主要由初始化與抽樣順序決定，而不是語料中的證據——這就是 1,000 篇摘要的限制。</p>`;
    } catch (e) { err(box, e); }
  }

  /* ------------------------------------------------------------------ *
   * wiring
   * ------------------------------------------------------------------ */
  $('#vb-go').addEventListener('click', loadVocab);
  $('#ex-go').addEventListener('click', genExamples);
  $('#ex-text').addEventListener('keydown', e => { if (e.key === 'Enter') genExamples(); });
  $('#nn-go').addEventListener('click', compare);
  $('#nn-words').addEventListener('keydown', e => { if (e.key === 'Enter') compare(); });
  $('#w2v-sim-btn').addEventListener('click', similar);
  $('#w2v-word').addEventListener('keydown', e => { if (e.key === 'Enter') similar(); });
  $('#sim-model').addEventListener('change', similar);
  $('#w2v-ana-btn').addEventListener('click', analogy);
  $('#ana-model').addEventListener('change', analogy);
  $('#pr-go').addEventListener('click', pairs);
  $('#pr-groups').addEventListener('keydown', e => { if (e.key === 'Enter') pairs(); });
  $('#pr-model').addEventListener('change', pairs);
  $('#map-go').addEventListener('click', maps);
  $('#map-seeds').addEventListener('keydown', e => { if (e.key === 'Enter') maps(); });
  $('#map-model').addEventListener('change', maps);
  $('#ev-go').addEventListener('click', () => { evaluate(); overlap(); });
  $('#ov-go').addEventListener('click', overlap);
  $$('#w2v-nav a').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $(a.getAttribute('href')).scrollIntoView({ behavior: 'smooth', block: 'start' }); }));
  let started = false;
  document.addEventListener('tab', e => {
    if (e.detail !== 'w2v') return;
    status();
    if (!started) { started = true; loadVocab(); genExamples(); }
  });
})();
