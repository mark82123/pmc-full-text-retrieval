# Project #3 — Term Weighting 與向量空間檢索（TF-IDF）

作者：胡元楨（P77141105）　課程：AI Information Retrieval（人工智慧資訊檢索）　繳交：2026-10-27

所有程式皆為純 Python 3 標準函式庫自行實作：`ir/tfidf.py`（權重與向量空間模型）、`ir/evaluation.py`（評估指標）、
`ir/relevance.py`（相關性判斷）、前端 `static/tfidf.js`（分頁「TF-IDF 檢索」，`/#vsm`）、CLI `python3 cli.py vsm …`。
文中每個數字都可由 CLI 重現（指令附在各節），單元測試在 `tests/test_project3.py`。

---

## 0. 資料集

| 項目 | 數值 |
|---|---|
| 集合 | PubMed 查詢 *GLP-1*，最新 3,000 篇英文摘要（`data/vsm/pubmed_glp1_3000.jsonl`，`efetch` 下載，2026-10-26） |
| 每筆記錄 | 標題、摘要段落、期刊、年份、作者、作者關鍵詞、**MeSH 標題**（descriptor、major-topic 旗標、qualifiers）、文獻類型 |
| 標籤覆蓋 | 1,704 筆有 MeSH（56.8%）、2,534 筆有關鍵詞（84.5%）、2,837 筆至少有一種（94.6%） |
| 文字 | 標題 + 摘要；855,731 tokens → 634,186 索引詞（停用詞移除 25.9%）；每篇 24–672 個索引詞，平均 211.4 |
| 詞彙表 | 15,823 個詞幹；7,122 個（45%）只出現在一篇文件（idf = ln 3000 = 8.01）；只有 42 個詞出現在 ≥ 1,000 篇 |

老師建議的 GLP-1 約 3,000 篇即此集合。Project #2 的 1,000 篇集合保留不動（`data/`），本專案的集合放在 `data/vsm/`，主索引與 Zipf / word2vec 分析不受影響。
`python3 cli.py vsm info`

---

## 任務一：文字前處理與詞項統計

前處理沿用 Project #1 的流程：tokenise（字母/數字連續片段；連字號分割，`GLP-1 → glp, 1`）→ 小寫 → 停用詞（Snowball 174 字 + 17 個生醫補充）→ Porter stemming。
對每篇文件 *d* 記錄 tf(t,d)、索引詞數 |d|、max tf(d)；對每個詞 t 記錄 cf、df、idf = ln(N/df)、prob-idf = max(0, ln((N−df)/df))。

| word | term | cf | df | df/N | idf | prob-idf | max tf | 1+ln(max tf) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| glp | glp | 9,842 | 2,333 | 77.8% | 0.252 | 0.000 | 19 | 3.94 |
| receptor | receptor | 4,889 | 2,422 | 80.7% | 0.214 | 0.000 | 19 | 3.94 |
| agonist | agonist | 4,510 | 2,335 | 77.8% | 0.251 | 0.000 | 19 | 3.94 |
| diabetes | diabet | 3,756 | 1,617 | 53.9% | 0.618 | 0.000 | 13 | 3.56 |
| obesity | obes | 4,052 | 1,429 | 47.6% | 0.742 | 0.095 | 18 | 3.89 |
| semaglutide | semaglutid | 2,478 | 714 | 23.8% | 1.435 | 1.164 | 29 | 4.37 |
| tirzepatide | tirzepatid | 1,472 | 440 | 14.7% | 1.920 | 1.761 | 13 | 3.56 |
| microbiota | microbiota | 327 | 129 | 4.3% | 3.147 | 3.103 | 9 | 3.20 |
| apnea | apnea | 143 | 63 | 2.1% | 3.863 | 3.842 | 10 | 3.30 |
| pcos | pco | 139 | 36 | 1.2% | 4.423 | 4.411 | 9 | 3.20 |

`python3 cli.py vsm terms glp receptor agonist diabetes obesity semaglutide tirzepatide microbiota apnea pcos`

**觀察。** 在一個以「GLP-1」為主題的集合裡，glp / receptor / agonist 出現在八成的文件中，idf 只有 0.21–0.25（prob-idf 為 0）；它們在集合層面是「主題詞」，但對區分文件幾乎沒有幫助。
真正能區分文件的是 semaglutide、tirzepatide（藥物）與 apnea、pcos（適應症）這類 df 在 1–25% 的詞。

---

## 任務二：Term weighting 的實作

權重以 SMART 三元記法 `tf · df · norm` 表示，三個因素可獨立切換（`ir/tfidf.py`）：

| 因素 | 代號 | 公式 |
|---|---|---|
| tf | **n** natural | tf |
| | **l** logarithm | 1 + ln tf |
| | **a** augmented | 0.5 + 0.5 · tf / max tf(d) |
| | **b** boolean | 1 |
| df | **n** none | 1 |
| | **t** idf | ln(N / df) |
| | **p** prob idf | max(0, ln((N − df) / df)) |
| norm | **n** none | 1 |
| | **c** cosine | 1 / ‖d‖（L2 正規化） |

作業要求的五種皆在其中：原始 TF = `nnn`、Log TF = `lnn`、IDF（只用 idf）= `btn`、TF-IDF = `ntn` / `ltn`、文件向量正規化 = `…c`。
另有 `atc`、`lpc` 兩種「modified TF-IDF」，以及向量空間之外的 Okapi BM25（k1 = 1.2，b = 0.75）作為參考。
每個 scheme 的文件向量以倒排結構（term → {doc: weight}）儲存，第一次使用時建立並快取；3,000 篇約 0.13 秒，一次查詢不到 1 ms。

**同一篇文件在不同 scheme 下的向量**（PMID42233552，FLOW 試驗腎臟結果分析，537 tokens → 422 索引詞、161 個不同詞、max tf = 38）：

| scheme | 權重最高的詞（tf, df → w） | 首詞佔向量權重 | 前 5 詞佔比 |
|---|---|---:|---:|
| `nnn` raw TF | **0** (38, 1172 → 38.0)、1 (18)、95 (13)、cardiovascular (13)、vs (13) | 9.0% | 22.5% |
| `lnn` log TF | 0 (38 → 4.64)、1 (→ 3.89)、95、cardiovascular、vs (→ 3.56) | 1.9% | 7.8% |
| `ltn` log TF × IDF | **ascvd** (6, 29 → 12.95)、264 (2, 2 → 12.38)、654、331、336 (2, 3–4 → 11.2–11.7) | 1.8% | 8.3% |
| `ltc` log TF × IDF, L2 | 同 `ltn`，每個權重除以 ‖d‖ = 66.6（ascvd → 0.194） | 1.8% | 8.3% |

`python3 cli.py vsm doc PMID42233552 --schemes nnn,lnn,ltn,ltc --top 6`

raw TF 讓一篇含大量統計數字的摘要被「0」（來自 0.5、p < 0.05 的小數）主宰；log 把 38 次壓成 4.6；idf 再把 df 1,172 的「0」壓下去、把 df 29 的 ascvd 抬上來。
副作用也看得到：df 只有 2–4 的數字 token（264、654、331）因 idf ≈ 7 而排到前面——在 Project #2 的 CF–DF 圖中這些是「number zone」的詞；若做「modified TF-IDF」，移除純數字 token 是最直接的改良（本集合有 1,819 個純數字詞，佔詞彙表 11.5%）。

---

## 任務三：簡易搜尋引擎

1. 文件 → 向量：如上。
2. 查詢 → 同一詞彙空間的向量：相同前處理；預設 `nnc`（原始 query tf、不乘 idf、L2 正規化）。查詢端的正規化只縮放分數、不改變排名；idf 由文件端供應，因此「文件 scheme」就是實驗變因。
3. 相似度：score(q,d) = Σ_t w(t,q) · w(t,d)，只走查詢詞的 postings。文件端為 `c` 時這正是 cosine similarity（分數 ∈ [0,1]）；為 `n` 時是未正規化內積 = ‖d‖ · cos θ。
4. 由高至低排序（分數相同時以 PMID 決定，結果可重現）。
5. 顯示前 10 名：分數、標題、年份、長度，以及每個查詢詞的貢獻 w(t,q) × w(t,d)（堆疊長條）；可展開摘要、MeSH、關鍵詞。

```
$ python3 cli.py vsm search "polycystic ovary syndrome" --scheme ltc -k 3
query terms: polycystic->polycyst(df=38, idf=4.3689), ovary->ovari(df=48, idf=4.1352), syndrome->syndrom(df=400, idf=2.0149)
query vector (nnc): {'polycyst': 0.5774, 'ovari': 0.5774, 'syndrom': 0.5774}
243 documents with a non-zero score in 0.65 ms; scheme ltc = log TF × IDF, L2; query q10: P@10 = 0.90
#1  PMID42549142  score=0.30601  rel=1  len=30   Correction: Comparative analysis of glucagon-like peptide-1 receptor agonists, metformin, …
      ovari: 0.5774×0.2034=0.11745  polycyst: 0.5774×0.2005=0.11575  syndrom: 0.5774×0.1261=0.07282
#2  PMID42708062  score=0.30036  rel=1  len=124  Uncovering the Metabolic Origins of Polycystic Ovary Syndrome
#3  PMID42857939  score=0.26118  rel=1  len=240  Preconception incretin therapy reduces pregnancy and cardiometabolic outcomes in polycystic …
```

---

## 任務四：對照實驗（一次只改一個因素）

固定查詢與查詢端權重（`nnc`），只改文件端 scheme。overlap@10 與 Kendall τ 以第一個 scheme 為基準；mean length 為前 10 名的平均索引詞數（全集合平均 211）。
`python3 cli.py vsm compare "<query>" --schemes … -k 10`

**RQ1 raw vs log TF**（q01 *semaglutide weight loss in obesity*）

| scheme | overlap@10 vs nnn | τ | mean length | mean max tf | P@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `nnn` raw TF | 1.00 | 1.00 | 338.7 | 18.8 | 0.80 | 0.440 |
| `lnn` log TF | 0.30 | −0.21 | 274.8 | 10.7 | 1.00 | 0.851 |
| `ntn` raw TF × IDF | 0.70 | 0.39 | 349.7 | 19.3 | 0.80 | 0.494 |
| `ltn` log TF × IDF | 0.50 | −0.12 | 279.8 | 12.5 | 1.00 | 0.818 |

`nnn` 的第一名是一篇「weight」出現 25 次的量測方法學文章（不相關）；換成 log 後前 10 名有 7 篇不同，且全部相關。加 idf 對 raw TF 幫助很小（ntn 與 nnn 重疊 0.7，第一名不變）：這個查詢的四個詞 idf 都不高（0.74–1.44），tf 的線性成長仍然主導。

**RQ2 有無 IDF**（q03 *GLP-1 receptor agonists and cardiovascular outcomes*；glp / receptor / agonist 的 idf 僅 0.21–0.25，cardiovascular 1.49）

| scheme | overlap@10 vs lnn | τ | mean length | P@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `lnn` log TF | 1.00 | 1.00 | 338.8 | 0.60 | 0.479 |
| `ltn` log TF × IDF | 0.40 | 0.05 | 338.1 | 0.90 | 0.866 |
| `btn` IDF only | 0.10 | −0.32 | 232.3 | 0.60 | 0.649 |
| `ltc` log TF × IDF, L2 | 0.10 | −0.36 | 151.0 | 1.00 | 0.762 |

沒有 idf 時排名由「GLP-1 receptor agonist」重複次數最多的文件決定：第一名是腎臟結果的比較研究，第 8–10 名是酒精使用、毛髮脫落、睡眠呼吸中止（皆不相關），前 10 名只有 6 篇相關；有 idf 後 cardiovascular 一詞的權重是 glp 的 6 倍，前 10 名有 9 篇是心血管結果的文章（nDCG@10 0.479 → 0.866）。

**RQ4 L2 正規化**（q10 *polycystic ovary syndrome*）

| scheme | overlap@10 vs ltn | τ | mean length | P@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| `ltn` log TF × IDF | 1.00 | 1.00 | 230.3 | 1.00 | 0.775 |
| `ltc` log TF × IDF, L2 | 0.60 | −0.17 | 164.6 | 1.00 | 0.528 |
| `nnn` raw TF | 0.50 | 0.09 | 214.7 | 0.70 | 0.609 |
| `nnc` raw TF, L2 | 0.30 | −0.32 | 152.0 | 0.80 | 0.525 |

正規化後前 10 名平均長度從 230 降到 165，且一篇 30 個詞的更正啟事（*Correction: …*）竄到第一名：它只有標題，三個查詢詞就佔了向量的大半，cosine 自然最高。

---

## 任務五：搜尋結果評估

### 相關性判斷（`data/vsm/queries_glp1.json`）

資料集沒有現成的相關性標籤，依作業指示改用**文件主題標籤**：每篇記錄的 MeSH 標題（NLM 索引員指派，含 major-topic 旗標）與作者關鍵詞。
每個查詢定義為若干「概念群」（正規表示式清單）；文件的**每一群**至少命中一個標籤即為相關（grade 1），命中的 MeSH 中有 major topic 則為 grade 2。
判斷只比對標籤、不看文字，因此不偏向任何 weighting。介面與 CLI 可手動覆寫個別判斷（`manual`），存回同一檔案。

**人工判讀補充。** 標籤規則的主要盲點是「文字明明相關、標籤卻沒標」。為了量化並修正它，對每個查詢把六個 scheme（nnn、lnn、btn、ltn、ltc、bm25）的前 10 名做 pooling（417 個 query–document 配對），
其中被規則判為 0 的 118 篇由作者閱讀標題與摘要後逐篇判定（0 = 確認不相關，1 / 2 = 相關但標籤漏標），結果寫入 `manual`：**37 篇**（31%）被補為相關（14 篇 grade 2、23 篇 grade 1），其餘 81 篇維持不相關。
漏標最多的是 q03（5 篇心血管結果試驗 / 統合分析的 MeSH 尚未索引）、q09（7 篇以副作用為主題的文章只有 adverse events 這類通用關鍵詞）、q13 / q14（retatrutide OSA 試驗、成本效益分析各 4 篇完全沒有標籤）；q07、q15 的 pool 沒有漏標。
人工判讀只覆蓋被檢索到的文件，所以它提高 P@k、MRR、nDCG 的可信度，對 R@k 的分母幾乎沒有影響。

| id | 查詢 | 相關 | g2 / g1 | 經 MeSH | 只有關鍵詞 | 人工判讀 / 補相關 |
|---|---|---:|---:|---:|---:|---:|
| q01 | semaglutide weight loss in obesity | 318 | 189 / 129 | 232 | 85 | 4 / 1 |
| q02 | tirzepatide glycemic control in type 2 diabetes | 131 | 82 / 49 | 97 | 32 | 16 / 2 |
| q03 | GLP-1 receptor agonists and cardiovascular outcomes | 263 | 160 / 103 | 163 | 95 | 16 / 5 |
| q04 | chronic kidney disease and diabetic nephropathy | 198 | 91 / 107 | 109 | 86 | 6 / 3 |
| q05 | gut microbiota and bile acid metabolism | 132 | 48 / 84 | 69 | 61 | 2 / 2 |
| q06 | fatty liver disease MASLD NAFLD steatohepatitis | 124 | 66 / 58 | 68 | 55 | 2 / 1 |
| q07 | neuroprotection in Alzheimer's and Parkinson's disease | 143 | 66 / 77 | 71 | 72 | 0 / 0 |
| q08 | bariatric surgery versus medication for weight loss | 139 | 66 / 73 | 81 | 57 | 5 / 1 |
| q09 | gastrointestinal adverse events nausea vomiting | 68 | 26 / 42 | 32 | 29 | 19 / 7 |
| q10 | polycystic ovary syndrome | 45 | 19 / 26 | 26 | 18 | 8 / 1 |
| q11 | SGLT2 inhibitors combination therapy | 262 | 107 / 155 | 156 | 104 | 3 / 2 |
| q12 | insulin secretion from pancreatic beta cells | 102 | 56 / 46 | 66 | 32 | 8 / 4 |
| q13 | obstructive sleep apnea treatment | 46 | 20 / 26 | 20 | 22 | 8 / 4 |
| q14 | cost-effectiveness and health economics | 30 | 14 / 16 | 20 | 6 | 6 / 4 |
| q15 | type 1 diabetes adjunct therapy | 51 | 34 / 17 | 36 | 15 | 15 / 0 |

`python3 cli.py vsm queries`　`python3 cli.py vsm relevant q10 --scheme ltc`　`python3 cli.py vsm judge q03 PMID42365335 2`

**限制。**（1）163 篇（5.4%）沒有任何標籤，永遠不會被判為相關——因此另外提供 condensed-list 評估（先移除無標籤文件再計算）；
（2）最新的記錄 MeSH 索引尚未完成（只有 57% 有 MeSH），這些文件只能靠關鍵詞；（3）規則由作者撰寫，不是獨立評審員的判讀；
（4）標籤層級的「相關」比人工判讀寬鬆：一篇以 Obesity 為 major topic、提到 semaglutide 的文章會被判為 q01 相關，即使重點不在減重。
相關集因此偏大（30–318 篇），R@10 的絕對值很低，應看相對差異；（5）人工判讀只做了 pooling 到的 118 篇，且由作者一人判定，未被任何 scheme 排進前 10 的漏標文件仍未修正。

### 結果（15 個查詢的平均；`python3 cli.py vsm eval`）

| scheme | P@5 | P@10 | R@10 | R@20 | MRR | MAP | nDCG@10 | nDCG@20 | R-prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nnn` raw TF | 0.773 | 0.760 | 0.091 | 0.158 | 0.791 | 0.451 | 0.545 | 0.517 | 0.449 |
| `lnn` log TF | 0.947 | 0.900 | 0.106 | 0.192 | 0.933 | 0.574 | 0.656 | 0.629 | 0.553 |
| `btn` IDF only | 0.840 | 0.787 | 0.083 | 0.166 | 0.922 | 0.546 | 0.640 | 0.595 | 0.562 |
| `ntn` raw TF × IDF | 0.907 | 0.880 | 0.105 | 0.195 | 0.933 | 0.599 | 0.642 | 0.636 | 0.574 |
| `ltn` log TF × IDF | 0.960 | **0.960** | 0.112 | **0.205** | 0.933 | **0.661** | 0.692 | **0.665** | **0.634** |
| `nnc` raw TF, L2 | 0.880 | 0.827 | 0.100 | 0.180 | 0.883 | 0.487 | 0.603 | 0.561 | 0.477 |
| `lnc` log TF, L2 | 0.920 | 0.873 | 0.106 | 0.196 | 0.967 | 0.574 | 0.663 | 0.621 | 0.553 |
| `ntc` raw TF × IDF, L2 | 0.933 | 0.920 | 0.106 | 0.199 | 0.967 | 0.609 | 0.675 | 0.644 | 0.585 |
| `ltc` log TF × IDF, L2 | 0.960 | 0.940 | 0.111 | **0.205** | 0.956 | 0.650 | **0.698** | 0.663 | 0.622 |
| `atc` augmented TF × IDF, L2 | 0.880 | 0.873 | 0.101 | 0.181 | **1.000** | 0.603 | 0.668 | 0.615 | 0.588 |
| `lpc` log TF × prob IDF, L2 | 0.947 | 0.947 | 0.111 | 0.201 | 0.967 | 0.654 | 0.695 | 0.658 | 0.622 |
| `bm25` 參考 | **0.973** | **0.960** | **0.113** | **0.205** | 0.967 | 0.656 | 0.683 | 0.657 | 0.619 |

（含 118 篇人工判讀；只用標籤規則時各欄約低 0.03–0.11，MAP 的排序相同：ltn 0.615、bm25 0.610、lpc 0.607、ltc 0.602、nnn 0.421。）
Condensed lists（移除無標籤文件）各欄數值變動 ±0.01，排序不變（ltn MAP 0.661、ltc 0.653、bm25 0.656、nnn 0.453）。

2 × 2 × 2 因素格（MAP）：

| tf \ idf · norm | 無 idf、無正規化 | idf、無正規化 | 無 idf、L2 | idf、L2 |
|---|---:|---:|---:|---:|
| raw tf | 0.451 | 0.599 | 0.487 | 0.609 |
| log tf | 0.574 | 0.661 | 0.574 | 0.650 |

每一步的平均效果：log（vs raw）+0.06 ~ +0.12 MAP；idf（vs 無）+0.08 ~ +0.15；L2（vs 無）−0.01 ~ +0.04。

---

## 研究問題

### RQ1 原始 TF 與 Log TF 的差異如何影響詞項權重？

raw tf 讓出現 30 次的詞權重是出現 1 次的 30 倍；1 + ln tf 只給 4.4 倍（tf = 2 → 1.69、5 → 2.61、10 → 3.30、50 → 4.91）。
在任務二的文件裡，raw TF 讓「0」一個詞佔向量 9%、前 5 詞佔 22.5%；log TF 把前 5 詞壓到 7.8%，向量不再被少數高頻詞綁架。
檢索上 raw → log 是所有單一改動中效果最大的：MAP 0.451 → 0.574（+27%）、nDCG@10 0.545 → 0.656、MRR 0.791 → 0.933；加 idf 後仍有 0.599 → 0.661，加 L2 後 0.609 → 0.650；逐題 AP 15 題全部提高。
前 10 名的平均長度也由 339 降到 275（q01）：raw tf 偏好能把同一個詞重複很多次的長文件。**詞頻增加，詞的重要性不會等比例增加。**

### RQ2 為什麼 IDF 能降低常見詞的重要性、提高具辨識力詞項的權重？

idf = ln(N/df) 隨 df 單調遞減：df = 1 → 8.01，df = 100 → 3.40，df = 1,000 → 1.10，df = 2,333（glp）→ 0.25，df = N → 0。
一個詞若出現在大多數文件中，它對「這篇 vs 那篇」沒有資訊——對所有文件加上同樣的分數不改變排名，只會讓分數被它的 tf 噪音淹沒。
在 q03 裡，沒有 idf 時排名由「GLP-1 receptor agonist」出現最多次的文件決定，第一名談腎臟、第 8–10 名談酒精、毛髮脫落與睡眠呼吸中止，前 10 名只有 6 篇相關；加上 idf 後 cardiovascular（idf 1.49）的權重是 glp（0.25）的 6 倍，排名才轉向主題（P@10 0.60 → 0.90，nDCG@10 0.479 → 0.866）。
整體：lnn → ltn MAP 0.574 → 0.661（+15%），lnc → ltc 0.574 → 0.650；逐題 AP 15 題全部提高。只用 idf（btn，完全忽略 tf）就有 MAP 0.546、nDCG@10 0.640，與 log TF（0.574 / 0.656）相去不遠——在摘要這種短文本裡，「有沒有出現」幾乎與「出現幾次」同樣重要，這也是 prob-idf（lpc）與 BM25 的 tf 飽和能表現良好的原因。

### RQ3 相較於只用 TF 或只用 IDF，TF-IDF 是否產生較好的排名？

是。MAP：raw TF 0.451、log TF 0.574、IDF only 0.546、log TF × IDF 0.661（+ L2：0.650）；R-precision 同序（0.449 / 0.553 / 0.562 / 0.634）；nDCG@10 的前三名 ltc 0.698、lpc 0.695、ltn 0.692 全是 TF × IDF 的組合。
逐題看 AP：`ltn` 在 15 題中全部高於 raw TF、全部高於 IDF-only、全部高於 log TF，沒有例外。
但「只用其中一個因素」的差距比想像中小：IDF-only 與 log TF 的 MAP 只差 0.03，兩者都比 raw TF 好 0.10 以上——tf 的線性成長才是最大的錯誤來源；
而 q04（chronic kidney disease）raw TF 的 nDCG@10 反而最好（0.815 vs ltn 0.645）——該題的詞（kidney、disease、chronic）idf 低而重複率高，tf 的訊息正好有用。

### RQ4 文件向量是否需要 L2 normalization？正規化前後的排名有何差異？

**排名差異很大，但在本集合不一定更好。** ltn 與 ltc 的前 10 名平均只有 48% 重疊（Kendall τ ≈ 0）；前 10 名平均長度：ltn 262 → ltc 186 個詞（全部 12 個 scheme 中所有未正規化者 224–288，正規化者 165–205）。
未正規化的內積 = ‖d‖·cos θ，長文件只要多提幾次查詢詞就贏；正規化後只比方向，長短文件公平——但極短的文件（30 個詞的更正啟事、評論）因為沒有別的詞稀釋，cosine 反而最高（q10 第一名）。
指標上，ltn → ltc MAP 0.661 → 0.650（15 題中只有 6 題正規化後 AP 較高）、P@10 0.960 → 0.940，但 MRR 0.933 → 0.956、nDCG@10 0.692 → 0.698 略升（正規化把短而切題的綜述推到最前面）；raw tf 則一致受益：nnn → nnc MAP 0.451 → 0.487（11 題變好）。
原因是本集合的文件長度相當一致（摘要，平均 211、相關文件平均 217），長度偏差本來就小，正規化的收益被「短文件竄升」抵銷。
在長度懸殊的集合裡正規化就是必要的：把 3 篇 PMC 全文（3,221–4,358 個詞）混進 1,003 篇摘要後，10 個查詢的前 10 名共 100 個位置中，`nnn` 有 15、`ltn` 有 9 個被這 3 篇長文佔走，`nnc` / `ltc` / BM25 為 0（`python3 report/rq4_mixed_lengths.py`）。
BM25 的長度正規化（b = 0.75，介於無與完全之間）在兩種情況都是最好或並列最好的，說明「要不要正規化」其實是「正規化多少」的問題。

### RQ5 不同 weighting 對 Precision@k、MRR、nDCG 等指標有何影響？

| 指標 | 看什麼 | 最佳 scheme | 說明 |
|---|---|---|---|
| P@5 / P@10 | 前幾名有多少相關 | bm25 0.973 / 0.960、ltn 0.960 / 0.960 | 前段品質由 tf 壓縮 + idf 決定；正規化略降 P@10（短文件竄升） |
| R@10 / R@20 | 找回多少相關文件 | bm25 0.113 / 0.205、ltn 0.112 / 0.205 | 相關集大（30–318），R@k 絕對值低；相對排序與 MAP 一致 |
| MRR | 第一個相關文件多前面 | atc 1.000、bm25 / lnc / ntc / lpc 0.967 | 幾乎所有 scheme 第一名都相關，正規化者略好；raw TF 最差（0.791） |
| MAP / R-prec | 整個排名 | ltn 0.661 / 0.634 | 對 tf 壓縮與 idf 最敏感：nnn → ltn +47% |
| nDCG@10 | 前 10 名含等級 | ltc 0.698、lpc 0.695、ltn 0.692 | 重視 grade-2（major topic）文件；正規化後短而切題的綜述排前面，略勝未正規化 |

指標之間的差異告訴我們：raw TF 在所有指標都最差，log TF × IDF（ltn / ltc / lpc）在所有指標都在前四，但「最佳」隨指標與查詢改變（MAP 選 ltn、nDCG@10 選 ltc、MRR 選 atc；q04 偏好 raw TF，q15 偏好正規化，q09 全部都差——副作用是幾乎每篇試驗摘要都會提到的詞，標籤與人工判讀卻只算「以副作用為主題」的文章）。
**不同 weighting 方法的優劣，必須結合查詢、資料集與評估指標判斷。**

---

## 結論

1. 詞頻增加，重要性不等比例增加：log TF 是單一最有效的改動（MAP +27%，15 題全部提高）。
2. 出現在許多文件中的詞缺乏區分能力：在 GLP-1 集合中 glp / receptor / agonist 的 idf ≈ 0.25，IDF 讓排名由主題詞轉向查詢的辨識詞（MAP 再 +15%）。
3. 正規化改變排名（重疊不到一半），其必要性取決於集合的長度分布：摘要集合中收益接近零並有短文件竄升的副作用，混合全文時則不可或缺；BM25 的部分正規化兩者兼顧。
4. 結論須結合指標：MAP / R-prec 選 `ltn`，nDCG@10 選 `ltc`，P@5 / R@k 選 BM25，MRR 選 `atc`；標準的 `ltc` 在各指標都在前三但不是全部最高。
5. 相關性判斷：標籤規則 + 118 篇 pooling 人工判讀（37 篇補為相關）；人工判讀讓各指標上升 0.03–0.11，但沒有改變任何結論的方向。
6. 可重現：`python3 cli.py vsm eval`、`vsm compare`、`vsm doc`、`vsm relevant`、`vsm judge` 與 `tests/test_project3.py`（19 個測試）。

## 附錄：指標定義（`ir/evaluation.py`）

P@k = |前 k 名中相關| / k；R@k = |前 k 名中相關| / |相關|；RR = 1 / 第一個相關文件的名次，MRR 為查詢平均；
AP = Σ_{相關文件 d 的名次 r} P@r / |相關|，MAP 為查詢平均；R-prec = P@|相關|；
DCG@k = Σ_{i≤k} (2^{grade_i} − 1) / log₂(i + 1)，nDCG@k = DCG@k / 理想排序的 DCG@k；
overlap@k = |top-k(A) ∩ top-k(B)| / k；Kendall τ 以兩個 top-10 聯集中的文件、依完整排名計算。
