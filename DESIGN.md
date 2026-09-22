# 系統設計說明（Design Notes）

> 這份文件說明系統的架構、每個模組的職責、核心演算法的原理，以及設計上的取捨。
> 文中的 `檔案:行號` 都對應到實際程式碼，報告或 code review 時可以直接對照。

---

## 1. 整體架構

```
                ┌────────────────────────────────────────────────────┐
  data/*.xml ──▶│ pmc_parser.py  JATS XML → Document(title, abstract, │
                │                sections[], authors, journal, year…) │
                └───────────────┬────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────┐
                │ tokenizer.py   文字 → Token(text, norm, stem, pos)  │
                │   ├ 斷詞  (regex [^\W_]+)                           │
                │   ├ 轉小寫                                          │
                │   ├ stopwords.py  停用詞移除（stem = ""）            │
                │   └ porter.py     Porter 詞幹                       │
                └───────────────┬────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────┐
                │ index.py       InvertedIndex                        │
                │   postings[term][doc_id] = [pos, pos, …]            │
                │   docs[doc_id] = 長度、欄位區間、詞頻                │
                └───────────────┬────────────────────────────────────┘
                                ▼
   查詢字串 ──▶ search.py  lex → QueryParser(AST) → Searcher.evaluate │
                │   → {doc_id: Match(hits)} → score(BM25/TF-IDF)      │
                └───────────────┬────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────┐
                │ engine.py      snippet、反白、match map、文件檢視    │
                │ stats.py       文件 / 語料庫統計                    │
                │ sentence.py    規則式句子切分                       │
                └───────┬─────────────────────────────┬──────────────┘
                        ▼                             ▼
                  app.py (HTTP + JSON API)       cli.py (命令列)
                        ▼
                  static/ (index.html, app.js, style.css)
```

**資料流**：啟動時 `Engine.load()`（`ir/engine.py:43`）讀取 `data/` 下所有 XML，
每篇文件經 parser → tokenizer → index 三步驟進入倒排索引，同時計算統計值。
查詢時 `Engine.search()`（`ir/engine.py:137`）呼叫 `Searcher`
取得排序後的文件清單，再為每篇產生 snippet、反白與圖表資料，回傳 JSON 給前端。

---

## 2. 各模組職責

| 模組 | 職責 | 關鍵函式 |
|---|---|---|
| `ir/pmc_parser.py` | 解析 JATS XML | `parse_article` (165)、`_text_of` (92)、`_sections` (146) |
| `ir/tokenizer.py` | 斷詞 + 正規化 | `Tokenizer.normalize` (42)、`iter_tokens` (54) |
| `ir/porter.py` | Porter 詞幹演算法 | `stem` (98)、`_measure` (25) |
| `ir/stopwords.py` | 191 個停用詞 | `STOP_WORDS` |
| `ir/sentence.py` | 規則式句子切分 | `split_sentences` (79)、`_is_boundary` (51) |
| `ir/index.py` | 位置倒排索引 | `add_document` (57)、`remove_document` (87) |
| `ir/search.py` | 查詢語法解析、比對、評分 | `lex` (50)、`QueryParser` (109)、`evaluate` (289)、`score` (324)、`_phrase_matches` (251) |
| `ir/stats.py` | 統計指標 | `document_stats` (12)、`corpus_stats` (53) |
| `ir/engine.py` | 整合層 | `search` (137)、`document` (177)、`_snippet` (108) |
| `app.py` | HTTP 伺服器 | `Handler.do_GET` / `do_POST` |
| `cli.py` | 命令列 | `main` |
| `static/app.js` | 前端 | `doSearch`、`renderResults`、`openDoc`、`renderCorpus` |

---

## 3. 核心演算法

### 3.1 XML 解析（`ir/pmc_parser.py`）

PMC 文章是 JATS 格式：`<front>` 放書目資料、`<body>` 放內文、`<back>` 放參考文獻。

* `_text_of()` 以遞迴走訪節點收集文字，但**跳過** `<xref ref-type="bibr">`（引用編號）、
  `<table>`、`<ref-list>` 等（`_SKIP_TAGS`，第 15 行）。原因：這些內容會塞進大量無意義的
  數字與人名，影響 df/idf 與統計值。
* 移除引用後會留下 `[–]`、`[, ]` 這種殘骸與「common .」這種句號前的空白，
  `_clean()` 用兩個 regex 收拾乾淨（第 86 行）。
* `Document.units()`（第 47 行）把文件整理成有序的 `(欄位, 文字)` 清單：
  `title → abstract 段落 → (heading, body 段落)…`。**整個系統的位置流、欄位區間、
  句子編號都建立在這個順序上**，所以索引與顯示才會一致。

### 3.2 斷詞與正規化（`ir/tokenizer.py`）

* Token = 連續的字母或數字（regex `[^\W_]+`）。連字號、撇號都會切開，
  所以 `T-cell` → `t`, `cell`；`COVID-19` → `covid`, `19`。
  取捨：文獻中 hyphen 用法不一致（`T cell` / `T-cell`），拆開比對更穩健。
* `normalize()` 回傳 `""` 代表停用詞；**Token 仍保留位置編號**（`position`），
  只是不寫進索引。這讓片語查詢能處理「role of the gut」（見 3.5）。

### 3.3 Porter 詞幹演算法（`ir/porter.py`）

把單字視為 `[C](VC)^m[V]`，`m` 是「母音-子音」交替次數（`_measure`，第 25 行）。
五個步驟依序去字尾：

| 步驟 | 作用 | 例子 |
|---|---|---|
| 1a | 複數 | caresses→caress, ponies→poni, cats→cat |
| 1b | -ed / -ing，並修補結尾 | plastered→plaster, hopping→hop, conflated→conflat |
| 1c | y→i | happy→happi |
| 2 | 長字尾 → 短字尾（m>0） | relational→relate, digitizer→digitize |
| 3 | -icate/-ful/-ness…（m>0） | hopefulness→hope |
| 4 | 去掉 -ance/-ment/-ion…（m>1） | replacement→replac, adoption→adopt |
| 5 | 去尾 e、ll→l | probate→probat, controll→control |

實作重點：`_apply_rules()`（第 86 行）依 Porter 原文規定，
**只要字尾符合就結束該步驟，即使 m 條件不成立也不再試其他規則**。
`PorterStemmer` 類別加了快取，因為索引時同一個字會重複出現數千次。
測試 `tests/test_ir.py::PorterTests` 用 Porter 論文中的範例驗證。

### 3.4 倒排索引（`ir/index.py`）

```
postings = { "immunotherapi": { "PMC8007559": [12, 57, 88, …],
                                "PMC7026250": [3, 41, …] }, … }
docs["PMC8007559"] = IndexedDoc(length=3120, n_tokens=6566,
                                field_ranges={"title": (0, 9), "abstract": (9, 210), "body": (210, 6566)},
                                term_freq=Counter(...))
```

* `length` 是去停用詞後的索引詞數（BM25 的 dl），`n_tokens` 是含停用詞的總位置數（match map 用）。
* `field_ranges` 記錄各欄位在位置流的區間，`title:` 查詢就是過濾位置是否落在區間內
  （`Searcher._in_field`，`ir/search.py:229`）。
* `remove_document()` 會反向清掉 posting，所以可以在網頁上動態加減文件。

### 3.5 查詢解析與比對（`ir/search.py`）

**Lexer**（第 50 行）把字串切成 `LPAREN / RPAREN / AND / OR / NOT / TERM / PHRASE`。
`-term` 等同 `NOT term`；`title:xxx` 會拆出欄位。

**Parser**（第 109 行）是遞迴下降（recursive descent），文法：

```
or_expr  := and_expr ( "OR" and_expr )*
and_expr := not_expr ( ["AND"] not_expr )*      ← 相鄰兩個運算元用「預設運算子」連接
not_expr := "NOT" not_expr | atom
atom     := "(" or_expr ")" | PHRASE | TERM | PREFIX
```

因此優先順序是 NOT > AND > OR，`a OR b AND c` → `(a OR (b AND c))`。

**Evaluate**（第 289 行）把 AST 轉成 `{doc_id: Match}`：

* `TERM`：查 postings。
* `AND`：各子結果的 key 取交集，並合併命中位置（`Match.merge`）。
* `OR`：取聯集。
* `NOT`：全部文件減去子結果（純 NOT 沒有命中資訊，分數 0）。
* `PREFIX`：`terms_with_prefix()` 展開後逐一 OR 起來。
* `PHRASE`（第 251 行）：對片語斷詞後得到 `[(詞幹, 相對位置)]`，例如
  `"role of the gut"` → `[("role", 0), ("gut", 3)]`（of、the 是停用詞但佔位置）。
  對第一個詞的每個出現位置 `p`，檢查其他詞是否出現在 `p + 位移`。
  這就是課堂上的 positional intersection，只是用 set 查詢取代雙指標走訪。

### 3.6 評分（`ir/search.py:324`）

* **BM25**（預設）
  `score = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`，
  `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`，k1 = 1.5、b = 0.75。
  tf 越高分數越高但會飽和；長文件會被正規化。
* **TF-IDF**：`(1 + ln tf) · ln((N+1)/df) / √dl`，簡化版的 cosine 正規化。
* **count**：命中次數，等於純布林檢索再按命中數排。
* 若該詞出現在標題，該詞的分數 ×1.25（title boost）。

### 3.7 規則式句子切分（`ir/sentence.py`）

候選邊界 regex（第 41 行）：`[.!?]+` 或 `...` → 可有可無的右引號/括號 → 空白 → 看下一個字元。
`_is_boundary()`（第 51 行）依序判斷：

1. 下一個字元是小寫 → 不切（`et al. reported`、`i.e. the`）。
2. `?`、`!`、刪節號 → 切。
3. 句號前的詞在縮寫表 → 不切（`Fig.`、`Dr.`、`vs.`、`approx.`）。
4. 句號前的詞只有一個字母 → 不切（`E. coli`、`J. R. Smith`）。
5. 句號前的詞是短的帶點縮寫（`a.m.`、`u.s`）→ 不切。
6. 小數 `0.05` 因為句號後沒有空白，根本不會成為候選。

段落是逐段送進 `split_sentences()`，所以段落結尾與標題自然成為邊界。
`naive_split()` 保留作為 baseline，介面與統計同時顯示兩者。

### 3.8 呈現層（`ir/engine.py`）

* **Snippet**（第 108 行）：把文件切成句子（有快取），對每句用 `highlight_spans()` 找出
  詞幹符合查詢的字，以「不同查詢詞的數量 ×1000 + 命中總數」排序，取前兩句，
  依原文順序顯示並用 `<mark>` 反白。反白是在**顯示時重新對每個字取詞幹**比對，
  所以 `vaccines`、`vaccination` 都會被標出來。
* **Match map**（第 129 行）：把命中位置除以 `n_tokens` 分到 40 個 bin，
  前端畫成小長條，看得出命中集中在文件的哪一段。
* **文件檢視**（第 177 行）：逐 unit、逐句輸出，附上句子編號與命中數；
  前端用 CSS class 切換「顯示句界」「只看有命中的句子」。

---

## 4. 設計決策與取捨

| 決策 | 理由 |
|---|---|
| 只用 Python 標準函式庫 | 課堂 demo 環境不必安裝套件；所有 IR 元件都是自己實作，符合作業要求 |
| 自己寫 Porter 而不用 NLTK | 作業允許用公開演算法，但自行實作能說明每一步；用論文範例做單元測試確保正確 |
| 停用詞不進索引但保留位置 | 索引縮小約 35%，片語查詢仍能處理 `role of the gut` |
| 移除引用、表格、參考文獻 | 避免 `[12]`、作者姓名、表格數字污染索引與統計 |
| 連字號切開 | 生醫文獻 hyphen 不一致，切開後 `T cell` / `T-cell` 都能匹配 |
| 預設運算子 AND | 與一般搜尋引擎一致；使用者可切換成 OR |
| 標題加權 ×1.25 | 標題出現查詢詞通常表示主題相關，但不至於壓過內文大量命中 |
| 前端不用框架、不用 CDN | 離線也能 demo；圖表用 inline SVG 自己畫 |
| 索引存在記憶體、啟動時重建 | 30 篇文件 0.9 秒建完，不值得為持久化增加複雜度 |

---

## 5. 已知限制

* 詞幹是英文專用，中文或其他語言文獻不適用。
* 前綴萬用字元是線性掃描整個詞彙表（8 千詞沒問題，百萬詞需要 trie 或排序陣列）。
* 句子規則以英文生醫文獻為對象，縮寫表需視語料調整；`U.S.` 結尾的句子會被誤判為不切。
* 沒有拼字校正、同義詞（MeSH）擴展、鄰近查詢（`a NEAR/5 b`）。
* 大語料（數萬篇）需要把索引壓縮並存到磁碟，目前全在記憶體。

---

## 6. 課堂可能被問的問題

**Q：為什麼 `immunotherapy` 查得到 `immunotherapies`？**
兩者經 Porter 後都是 `immunotherapi`；索引與查詢用同一個 `Tokenizer.normalize()`。

**Q：片語查詢中的停用詞怎麼處理？**
斷詞時停用詞仍佔一個位置。`"role of the gut"` 轉成 `role@0, gut@3`，
比對時檢查 `gut` 是否出現在 `role` 的位置 +3。

**Q：BM25 與 TF-IDF 差在哪？可以現場示範嗎？**
切換 Scoring 下拉選單即可。BM25 的 tf 會飽和且有長度正規化，所以一篇短但集中的文章
可能贏過一篇很長、詞很多但密度低的文章；count 模式則單純比命中次數。

**Q：句數為什麼比 naive 少那麼多（6,809 vs 9,003）？**
生醫文獻大量出現 `Fig. 2`、`et al.`、`p < 0.05`、`E. coli`，naive 每個句號都切。
可以在文件檢視打開「Show sentence boundaries」逐句檢查。

**Q：NOT 單獨用會怎樣？**
`NOT cancer` 回傳所有不含 cancer 的文件，分數皆為 0（沒有正向命中），順序依文件 ID。

**Q：新增文件時索引怎麼更新？**
`InvertedIndex.add_document()` 把新文件的 posting 加進去；同 ID 會先 `remove_document()`
再加，並重算 `total_length`，所以 avgdl 立刻反映。

**Q：為什麼標題不算進 body 段落數，但算進句數與字數？**
統計以 `Document.units()` 為準：字數、句數涵蓋所有 unit（title/abstract/heading/body），
「paragraphs」只數 body 段落，讓兩者意義清楚。

---

## 7. 建議的 Demo 流程（約 8 分鐘）

1. `python3 cli.py analyze "Kim et al. reported p < 0.05 in Fig. 2A. The T-cells were activated."`
   — 展示斷詞、停用詞、詞幹、句子切分（naive 6 句 vs 規則式 3 句）。
2. 網頁搜尋 `cancer immunotherapy` — 看 query analysis（詞幹、df）、snippet 反白、match map、詞頻圖。
3. 改成 `"gut microbiome" AND diabetes`、`immun*`、`title:crispr`、`covid AND vaccine NOT pregnancy` — 展示查詢語法。
4. 切換 BM25 / TF-IDF / count，指出排名變化。
5. 點進一篇文件 — 統計 tiles、句界標記、只看命中句。
6. Corpus 分頁 — 語料統計、Zipf 圖、排序表格。
7. 上傳一篇新的 PMC XML（先從 PMC 網站下載）— 立即可搜尋。
8. `python3 -m unittest discover -s tests -v` — 19 個測試。

---

## 8. 自我驗證練習（幫助內化程式碼）

做過這幾件事，回答 code review 問題會很有把握：

1. 在 `ir/sentence.py` 的 `ABBREVIATIONS` 加一個縮寫（例如 `spp`），用 `cli.py analyze` 觀察差異。
2. 把 `ir/search.py:324` 的 `k1` 改成 0.5 或 3，再搜同一個查詢，看排名怎麼變。
3. 把 title boost 的 1.25 改成 1.0，觀察 `title:` 以外的查詢排名。
4. 在 `tests/test_ir.py` 加一個片語測試，例如 `"lung cancer"` 應只命中 D2。
5. 用 `python3 cli.py vocab immun` 看前綴展開了哪些詞，對照 `immun*` 的 term_hits。
