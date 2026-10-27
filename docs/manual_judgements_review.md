# Manual relevance judgements — review list

The 118 `manual` entries in `data/vsm/queries_glp1.json` were assigned on 2026-10-27 by the AI assistant (Claude) that helped build this project, by reading the title and abstract of every pooled top-10 document (six schemes: nnn, lnn, btn, ltn, ltc, bm25) that the label rule scored 0. The author has **not yet reviewed them one by one**; this file exists so that review can be done quickly.

Criteria used: **2** = the query topic is the main subject of the abstract (e.g. a trial whose primary outcome is the queried condition); **1** = the topic is substantively discussed but secondary; **0** = only mentioned in passing or not at all.

Totals: 37 upgraded to relevant (1 or 2), 81 confirmed 0.

How to change one: web tab TF-IDF 檢索 → section 3 → buttons 0 / 1 / 2 / × next to the result, or `python3 cli.py vsm judge <qid> <PMID> <grade>` (empty grade removes the override).

## q01  semaglutide weight loss in obesity

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 1 | PMID42251859 | Cagrilintide-semaglutide (CagriSema) versus semaglutide or cagrilintide in people with type 2 diabetes (REIMAGINE 2): a double-blind, randomised, controlled, phase 3 study | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID41307513 | Efficacy and safety of oral semaglutide as add-on therapy in poorly controlled type 2 diabetes on background SGLT2 inhibitors: a real-world, multi-center, retrospective, observational study (RYS2) | Minerva endocrinology (2026) | ☐ |
| 0 | PMID42767324 | Can obesity pharmacotherapy modify osteoarthritis? Interpreting pain relief, weight-loss quality, and structural outcomes | Osteoarthritis and cartilage (2026) | ☐ |
| 0 | PMID42821214 | Weight Loss Measurements Under the Microscope: How Type and Baseline Definition Shape Clinical Response After Metabolic-Bariatric Surgery - A 2-Year Follow-up Analysis of 17,568 Patients | Obesity surgery (2026) | ☐ |
## q02  tirzepatide glycemic control in type 2 diabetes

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 1 | PMID42394981 | Efficacy and safety of incretin-based therapies in patients with type 2 diabetes mellitus: a network meta-analysis based on clinical trials | Frontiers in pharmacology (2026) | ☐ |
| 1 | PMID42534427 | Beyond glycemic control: clinical cardiovascular effects of tirzepatide-a narrative review | Therapeutic advances in endocrinology and metabolism (2026) | ☐ |
| 0 | PMID41883513 | Tirzepatide as adjunct therapy in patients with type 1 diabetes: a systematic review and meta-analysis | Journal of diabetes and metabolic disorders (2026) | ☐ |
| 0 | PMID42030105 | Whey Protein Ingestion Stimulates Glucagon Secretion and Raises Blood Glucose Levels in Adults With Type 1 Diabetes | Diabetes (2026) | ☐ |
| 0 | PMID42035781 | Continuous glucose monitoring versus self-monitoring of blood glucose in individuals with type 2 diabetes: a randomised, multicentre, open-label, superiority trial | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42097623 | Effects of Semaglutide on Weight and Insulin Requirements in Two Adolescents With Type 1 Diabetes | Pediatrics (2026) | ☐ |
| 0 | PMID42234428 | HRS-7535 for Type 2 Diabetes Inadequately Controlled With Metformin: A Randomized Clinical Trial | JAMA network open (2026) | ☐ |
| 0 | PMID42250575 | Efficacy and safety of retatrutide, a GIP, GLP-1, and glucagon receptor agonist, in people with type 2 diabetes and inadequate glycaemic control with diet and exercise (TRANSCEND-T2D-1): a double-blind, randomised, phase 3 trial | Lancet (London, England) (2026) | ☐ |
| 0 | PMID42251769 | Orforglipron Added to Titrated Insulin Glargine in Type 2 Diabetes: The ACHIEVE-5 Randomized Clinical Trial | JAMA (2026) | ☐ |
| 0 | PMID42251856 | Cagrilintide-semaglutide (CagriSema) as an add-on to basal insulin in adults with type 2 diabetes (REIMAGINE 3): a randomised, double-blind, placebo-controlled, multicentre, phase 3 study | Lancet (London, England) (2026) | ☐ |
| 0 | PMID42251859 | Cagrilintide-semaglutide (CagriSema) versus semaglutide or cagrilintide in people with type 2 diabetes (REIMAGINE 2): a double-blind, randomised, controlled, phase 3 study | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42251860 | Efficacy and safety of once-weekly cagrilintide-semaglutide (CagriSema) in adults with type 2 diabetes inadequately controlled on diet and exercise (REIMAGINE 1): a randomised, double-blind, placebo-controlled, phase 3a study | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42639616 | When Weight Loss Turns Dangerous: A Case Report of Tirzepatide-Induced Ketoacidosis in a Non-diabetic Woman | Cureus (2026) | ☐ |
| 0 | PMID42810372 | Retatrutide in adults with obesity and type 2 diabetes (TRIUMPH-2): a double-blind, parallel-group, randomised, placebo-controlled, phase 3 trial | Lancet (London, England) (2026) | ☐ |
| 0 | PMID42862661 | GLP-1 Receptor Agonists and Insulin Versus Insulin for Perioperative Glycemic Control in Patients with Type 2 Diabetes Undergoing Cardiac Surgery: A Systematic Review | Annals of cardiac anaesthesia (2026) | ☐ |
| 0 | PMID42898168 | Effects of intensive insulin therapy combined with non-insulin antihyperglycemic agents on glycemic control, variability, and metabolic outcomes in patients with type 2 diabetes mellitus: a systematic review and meta-analysis | Frontiers in endocrinology (2026) | ☐ |
## q03  GLP-1 receptor agonists and cardiovascular outcomes

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID41903177 | Cardiorenal Outcomes With Tirzepatide Compared With Dulaglutide in Patients With Diabetes and Cardiovascular Disease: A Post Hoc Analysis of the SURPASS-CVOT Randomized Clinical Trial | JAMA cardiology (2026) | ☐ |
| 2 | PMID42365335 | Comparative cardiovascular efficacy of antidiabetic therapies in type 2 diabetes: a systematic review and network meta-analysis of randomized trials | Diabetology & metabolic syndrome (2026) | ☐ |
| 2 | PMID42624784 | Not Just for Diabetes: The Next Frontier for GLP-1 Agonists | Journal of the American Board of Family Medicine : JABFM (2026) | ☐ |
| 2 | PMID42815506 | Cardiovascular safety of orforglipron versus insulin glargine in adults with type 2 diabetes at increased cardiovascular risk (ACHIEVE-4): a phase 3, event-driven, randomised, open-label, non-inferiority, active comparator trial | Lancet (London, England) (2026) | ☐ |
| 1 | PMID42681751 | Incretin-Based Therapies as Cardiometabolic Interventions: Outcome Hierarchy, Receptor-Network Pharmacology and Phenotype-Based Clinical Positioning | Diabetes, obesity & metabolism (2026) | ☐ |
| 0 | PMID41355396 | Association of Baseline Comorbidities With First-Year Adherence to GLP-1 Receptor Agonists in Patients With Diabetes or Obesity: A Retrospective Cohort Study | The Annals of pharmacotherapy (2026) | ☐ |
| 0 | PMID41569401 | Targeting Vascular Calcification: Novel Insights into Molecular Pathways and Clinical Interventions | Cardiovascular drugs and therapy (2026) | ☐ |
| 0 | PMID41665869 | Renal and urothelial cancer risks with SGLT2 inhibitors vs GLP-1 receptor agonists in type 2 diabetes: a target trial emulation | Nephrology, dialysis, transplantation : official publication of the European Dialysis and Transplant Association - European Renal Association (2026) | ☐ |
| 0 | PMID41665888 | GLP-1 receptor agonists and the risk of fragility fractures in older adults with type 2 diabetes | The Journal of clinical endocrinology and metabolism (2026) | ☐ |
| 0 | PMID42302979 | Comparison of Specific Glucagon-Like Peptide-1 Receptor Agonists on Kidney Outcomes Among Patients With Type 2 Diabetes | American journal of kidney diseases : the official journal of the National Kidney Foundation (2026) | ☐ |
| 0 | PMID42309434 | Adjunctive GLP1 Receptor Agonists in Patients With Inflammatory Bowel Diseases and Obesity and/or Diabetes: A Target Trial Emulation | Clinical gastroenterology and hepatology : the official clinical practice journal of the American Gastroenterological Association (2026) | ☐ |
| 0 | PMID42456704 | Association of GLP-1 receptor agonists with alcohol use disorder-related and substance use disorder-related hospital admissions during treatment and after discontinuation: a Swedish register-based within-individual observational study | The lancet. Psychiatry (2026) | ☐ |
| 0 | PMID42486607 | Risk of hair loss associated with glucagon-like peptide-1 receptor agonists in adults with type 2 diabetes: target trial emulation | BMJ (Clinical research ed.) (2026) | ☐ |
| 0 | PMID42676363 | Efficacy of SGLT2 inhibitors, GLP-1 receptor agonists, and aerobic exercise for moderate-to-severe obstructive sleep apnea in overweight or obese patients: a network meta-analysis | Frontiers in endocrinology (2026) | ☐ |
| 0 | PMID42705550 | Impact of GLP-1 dose intensity on perioperative and long-term fusion outcomes following ACDF | The spine journal : official journal of the North American Spine Society (2026) | ☐ |
| 0 | PMID42789287 | GLP-1 Receptor Agonist Coverage for Obesity in Medicaid and Total GLP-1 Receptor Agonist Utilization | JAMA health forum (2026) | ☐ |
## q04  chronic kidney disease and diabetic nephropathy

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 1 | PMID42470502 | Combination cardiometabolic therapy in type 2 diabetes: optimizing SGLT2 inhibitor and GLP‑1 receptor agonist use | Current atherosclerosis reports (2026) | ☐ |
| 1 | PMID42608342 | Therapies for Cardiovascular-Kidney-Liver-Metabolic Syndrome: Reappraisal of Fibrates | Diabetes, obesity & metabolism (2026) | ☐ |
| 1 | PMID42705609 | Impact of renal lipid droplet dynamics on tubular health | Journal of lipid research (2026) | ☐ |
| 0 | PMID42292743 | Mesenchymal Stem Cell Therapy for Type 2 Diabetes: Synergistic β-Cell Regeneration, Immune Modulation, and Exosome-Mediated Glucose Homeostasis | Stem cells international (2026) | ☐ |
| 0 | PMID42819943 | Cardiovascular protection beyond glucose lowering: GLP-1 receptor agonists and dual GIP/GLP-1 receptor agonism in contemporary cardiology | Frontiers in cardiovascular medicine (2026) | ☐ |
| 0 | PMID42918905 | GLP-1-based therapies in cardiovascular disease: outcomes, mechanisms, and emerging clinical applications | European journal of clinical pharmacology (2026) | ☐ |
## q05  gut microbiota and bile acid metabolism

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42294227 | Liraglutide and Dapagliflozin Synergistically Reshape Gut Microbiota and Metabolic Profiles to Ameliorate Type‑2 Diabetes in Mice | ACS omega (2026) | ☐ |
| 2 | PMID42499664 | Tirzepatide and the gut microbiota-obesity axis: metabolic mechanisms and therapeutic perspectives | Frontiers in microbiology (2026) | ☐ |
## q06  fatty liver disease MASLD NAFLD steatohepatitis

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42776095 | AGA Clinical Practice Update on the Evaluation and Management of Metabolic Dysfunction- and Alcohol-Associated Liver Disease: Expert Review | Clinical gastroenterology and hepatology : the official clinical practice journal of the American Gastroenterological Association (2026) | ☐ |
| 0 | PMID42056219 | Short-, medium-, versus long-chain fatty acids: mechanisms of immunomodulation and disease pathogenesis | Cellular & molecular immunology (2026) | ☐ |
## q08  bariatric surgery versus medication for weight loss

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 1 | PMID42781681 | Improving Patient Outcomes by Bridging Bariatric and Body Contouring Surgery with Comprehensive Prehabilitation | Plastic and reconstructive surgery. Global open (2026) | ☐ |
| 0 | PMID40549581 | Medications for Weight Loss and MASLD: A National Survey of Hepatology and Gastroenterology Provider Practices, Attitudes, and Knowledge Before Resmetirom | Journal of clinical gastroenterology (2026) | ☐ |
| 0 | PMID40580049 | The quantity, quality and findings of network meta-analyses evaluating the effectiveness of GLP-1 RAs for weight loss: a scoping review | Health technology assessment (Winchester, England) (2026) | ☐ |
| 0 | PMID42105046 | Semaglutide and Cardiovascular Outcomes in People with Type 2 Diabetes: A SUSTAIN-6 Post Hoc Analysis by Weight Loss Category | Diabetes therapy : research, treatment and education of diabetes and related disorders (2026) | ☐ |
| 0 | PMID42579223 | Skin Impacts and Tradeoffs of GLP-1 Therapy: Improved Patient-Reported Outcomes of Inflammatory Skin Disease in the Era of "Ozempic Face" | Dermatology and therapy (2026) | ☐ |
## q09  gastrointestinal adverse events nausea vomiting

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42544311 | Adverse Events Associated with Incretin-Based Therapies: A Narrative Review on Mechanisms, Clinical Management, and Risk Mitigation | Drug design, development and therapy (2026) | ☐ |
| 2 | PMID42622003 | Severe Gastrointestinal Intolerance After Resuming Maintenance-Dose Semaglutide Following Ramadan-Related Treatment Interruption: A Case Report | Cureus (2026) | ☐ |
| 2 | PMID42906844 | The Impact of Glucagon-Like Peptide-1 Receptor Agonists on Post-Anesthetic Pulmonary and Gastrointestinal Adverse Events: A Target Trial Emulation Using Multinational Database | The American journal of medicine (2026) | ☐ |
| 1 | PMID41991422 | Approved weight loss drugs for obesity with a thorough emphasis on GLP-1 agonist medications: A systematic review | Disease-a-month : DM (2026) | ☐ |
| 1 | PMID42334501 | Potential oral and maxillofacial adverse events associated with glucagon-like peptide-1 receptor agonist administration: a narrative review | Quintessence international (Berlin, Germany : 1985) (2026) | ☐ |
| 1 | PMID42822760 | Survodutide for the treatment of obesity: Mechanistic rationale, clinical evidence, and remaining uncertainties | Contemporary clinical trials (2026) | ☐ |
| 1 | PMID42863671 | Characterizing the adverse event profile of semaglutide: A comprehensive U.S. food and drug administration adverse event reporting system-based pharmacovigilance study | Clinical nutrition ESPEN (2026) | ☐ |
| 0 | PMID41644273 | Cardiorenal protective effects of glucagon-like peptide-1 receptor agonists in chronic kidney disease: a systematic review and meta-analysis | Renal failure (2026) | ☐ |
| 0 | PMID41848415 | Efficacy and safety of glucagon-like peptide-1 (GLP-1) agonists in pediatric obesity: A systematic review and meta-analysis | Saudi journal of gastroenterology : official journal of the Saudi Gastroenterology Association (2026) | ☐ |
| 0 | PMID42259343 | Elecoglipron, an oral small molecule GLP-1 receptor agonist in adults with type 2 diabetes (SOLSTICE): a multicentre, phase 2b, randomised, placebo-controlled trial | Lancet (London, England) (2026) | ☐ |
| 0 | PMID42316363 | Pulmonary adverse events associated with GLP-1 receptor agonists: a systematic review of respiratory safety signals | Cardiovascular diabetology. Endocrinology reports (2026) | ☐ |
| 0 | PMID42365860 | GLP-1 receptor agonists in pediatric obesity and diabetes: a systematic review of efficacy, metabolic effects, and safety | Diabetes research and clinical practice (2026) | ☐ |
| 0 | PMID42422116 | Efficacy and safety of GLP-1RA on cardio-metabolic outcomes in overweight or obese Chinese adults: a systematic review and meta-analysis | Journal of diabetes and metabolic disorders (2026) | ☐ |
| 0 | PMID42432163 | Glucagon-like peptide-1 agonists in Parkinson's disease: a meta-analysis | Neurological sciences : official journal of the Italian Neurological Society and of the Italian Society of Clinical Neurophysiology (2026) | ☐ |
| 0 | PMID42483679 | Investigation into the efficacy and safety profile of oral small-molecule GLP-1 receptor agonists in type 2 diabetes and obesity: a systematic review and meta-analysis | Frontiers in endocrinology (2026) | ☐ |
| 0 | PMID42536519 | Effectiveness & safety of semaglutide in weight reduction in obese nondiabetic patients: A systematic review and meta-analysis | Medicine (2026) | ☐ |
| 0 | PMID42692277 | Glucagon-like peptide-1 receptor agonist-associated injection site and dermatologic reactions | Clinics in dermatology (2026) | ☐ |
| 0 | PMID42744082 | Efficacy and safety of avexitide for treatment of hypoglycemia following bariatric and other upper gastrointestinal surgeries: a phase 2b study | Metabolism: clinical and experimental (2026) | ☐ |
| 0 | PMID42815506 | Cardiovascular safety of orforglipron versus insulin glargine in adults with type 2 diabetes at increased cardiovascular risk (ACHIEVE-4): a phase 3, event-driven, randomised, open-label, non-inferiority, active comparator trial | Lancet (London, England) (2026) | ☐ |
## q10  polycystic ovary syndrome

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42880183 | Polycystic ovary syndrome: An insight into pathophysiology, current management, and emerging therapeutic approaches | Current opinion in pharmacology (2026) | ☐ |
| 0 | PMID41885132 | Safety of GLP-1 and Dual GLP-1/GIP Receptor Agonists in Preconception, Pregnancy, and Lactation: A Systematic Review of Maternal, Fetal, and Neonatal Outcomes | Diabetes, obesity & metabolism (2026) | ☐ |
| 0 | PMID41913544 | Beyond Weight Loss: Metabolic and Immunologic Power of GLP-1 Agonists in Plastic Surgery | Aesthetic surgery journal (2026) | ☐ |
| 0 | PMID42037238 | Regulatory cycles of orexin and glucagon-Like peptide-1 in postviral syndromes | Endocrine reviews (2026) | ☐ |
| 0 | PMID42082824 | Mitochondrial Function and Dysfunction in Female Fertility: Biological Mechanisms, Genetic Determinants, and Therapeutic Opportunities - A Review | Reproductive sciences (Thousand Oaks, Calif.) (2026) | ☐ |
| 0 | PMID42208956 | Beyond weight loss: multisystem benefits of obesity medications | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42411248 | Updates in management of cardiorenal syndrome | Current opinion in nephrology and hypertension (2026) | ☐ |
| 0 | PMID42597412 | Evaluating the use of GLP-1 receptor agonists in Wolfram syndrome patients | Frontiers in endocrinology (2026) | ☐ |
## q11  SGLT2 inhibitors combination therapy

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42500549 | Combination GLP-1 receptor agonist and SGLT2 inhibitor therapy versus GLP-1 receptor agonist monotherapy on arrhythmia outcomes in non-diabetic obese patients: A real-world time-dependent analysis | International journal of cardiology. Cardiovascular risk and prevention (2026) | ☐ |
| 1 | PMID42647034 | Glucagon-Like Peptide 1 Receptor Agonists, Mortality, and Cardiovascular Outcomes in Serious Mental Illness | JAMA psychiatry (2026) | ☐ |
| 0 | PMID42086460 | Adherence and Persistence Among Individuals Living With Type 2 Diabetes and Using a Free Combination of Basal Insulin and an Injectable Glucagon-Like Peptide-1 Receptor Agonist Versus a Fixed-Ratio Combination Therapy: A Japanese Database Study | Diabetes, obesity & metabolism (2026) | ☐ |
## q12  insulin secretion from pancreatic beta cells

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42526996 | GLP-1 Receptor Agonists May Reduce Insulin Requirements in New-Onset Type 1 Diabetes | Endocrinology and metabolism clinics of North America (2026) | ☐ |
| 2 | PMID42737905 | Combined Exendin-4 and Rituximab Treatment Improves β-Cell Function and Delays Disease Progression in NOD Mice | Biology (2026) | ☐ |
| 1 | PMID42062836 | Gut-Derived GLP-1 Released by Rare Sugar d-Allulose Cooperates With Insulin to Activate Left-Sided Vagal Afferents and Enhance Insulin Sensitivity | Diabetes (2026) | ☐ |
| 1 | PMID42736532 | An illustrative summary of receptor-mediated mechanisms involved in the therapeutic regulation of type 2 diabetes | Hormones (Athens, Greece) (2026) | ☐ |
| 0 | PMID42697752 | Use of fixed-ratio combination of basal insulin and glucagon-like peptide-1 receptor agonists (GLP-1 RAs) in Asians with type 2 diabetes | Primary care diabetes (2026) | ☐ |
| 0 | PMID42728365 | Paediatric proteomic signatures of cardiometabolic disease-associated traits predict adult disease outcomes | Nature metabolism (2026) | ☐ |
| 0 | PMID42758304 | Unimolecular GLP-1/APJ receptor co-agonism improves metabolism and dyslipidaemia as well as normalising hepatic triglyceride and aminotransferase elevation in high-fat-fed mice | Diabetologia (2026) | ☐ |
| 0 | PMID42784708 | The Intertwined Threat: How Diabetes Impacts Neurological Disorders - A Review Article | Acta neurologica Taiwanica (2026) | ☐ |
## q13  obstructive sleep apnea treatment

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42378200 | Endotypic Trait Responses to 12-Month Tirzepatide Treatment for an Obese Patient with Obstructive Sleep Apnea: A Case Report | Obesity facts (2026) | ☐ |
| 2 | PMID42814954 | Retatrutide, a Triple Hormone Receptor Agonist, for Treatment of Obesity | The New England journal of medicine (2026) | ☐ |
| 2 | PMID42904878 | Retatrutide Once Weekly for Treatment of Obstructive Sleep Apnea | The New England journal of medicine (2026) | ☐ |
| 1 | PMID42823495 | Obesity, sleep apnea and cancer revisited | International journal of obesity (2005) (2026) | ☐ |
| 0 | PMID41989133 | Complex clinical encounter series: glucagon-like peptide-1 receptor agonists-induced weight loss: are we paying attention to bone health? | Journal of bone and mineral research : the official journal of the American Society for Bone and Mineral Research (2026) | ☐ |
| 0 | PMID42054048 | The National Psoriasis Foundation Primer on GLP-1 Receptor Agonists in Psoriasis: A Review | JAMA dermatology (2026) | ☐ |
| 0 | PMID42402270 | Total care strategies in atrial fibrillation practice: Lifestyle intervention and risk factor management | Journal of cardiology (2026) | ☐ |
| 0 | PMID42444567 | Medical treatments for obesity: what does the future have in store? | The Journal of clinical endocrinology and metabolism (2026) | ☐ |
## q14  cost-effectiveness and health economics

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 2 | PMID42736030 | Real-World Cost and Utilisation Burden of Diagnosed Metabolic Dysfunction-Associated Steatotic Liver Disease in Patients With Type 2 Diabetes in the United States: A Retrospective Cohort Study | Diabetes, obesity & metabolism (2026) | ☐ |
| 2 | PMID42808550 | Real-world treatment patterns and cost per patient achieving treatment targets among tirzepatide and semaglutide initiators in US patients with type 2 diabetes | Journal of managed care & specialty pharmacy (2026) | ☐ |
| 2 | PMID42890581 | The Cost of Inaction in Treating Obesity in the U.S | Diabetes, obesity, and cardiometabolic CARE (2026) | ☐ |
| 2 | PMID42894545 | Economic evaluation of tirzepatide, semaglutide, and liraglutide in the treatment of obese and overweight patients without diabetes: a short-term cost-effectiveness analysis in Iran | Journal of diabetes and metabolic disorders (2026) | ☐ |
| 0 | PMID42008388 | Harnessing Herbal Power: A Systematic Review of Phytopharmaceuticals in Type 2 Diabetes Management | Journal of the American Nutrition Association (2026) | ☐ |
| 0 | PMID42372278 | Leveraging Real-World Evidence to Inform Regulatory, Clinical, and Coverage Decisions Related to Glucagon-like Peptide-1-Based Therapies: Synopsis of a National Institute of Diabetes and Digestive and Kidney Diseases Workshop | Annals of internal medicine (2026) | ☐ |
## q15  type 1 diabetes adjunct therapy

| grade | PMID | title | journal (year) | reviewed |
|---|---|---|---|---|
| 0 | PMID41795036 | Endoplasmic reticulum-mitochondria contact sites are signalling hubs connecting nutrient sensing and GLP-1 secretion in L cells of the mouse gut: from physiology to obesity and type 2 diabetes | Diabetologia (2026) | ☐ |
| 0 | PMID41999297 | GLP-1 receptor agonists in stroke prevention: a narrative review on emerging therapeutic frontiers | Annals of medicine (2026) | ☐ |
| 0 | PMID42035781 | Continuous glucose monitoring versus self-monitoring of blood glucose in individuals with type 2 diabetes: a randomised, multicentre, open-label, superiority trial | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42054048 | The National Psoriasis Foundation Primer on GLP-1 Receptor Agonists in Psoriasis: A Review | JAMA dermatology (2026) | ☐ |
| 0 | PMID42217111 | Defining the Roles of SGLT2 Inhibitors and GLP-1 Receptor Agonists in the Management of Chronic Kidney Disease in Adults with Type 2 Diabetes With or Without Overweight/Obesity: An International Delphi Consensus | Advances in therapy (2026) | ☐ |
| 0 | PMID42219586 | GLP-1 RAs in Substance Use Disorders: Emerging Evidence and Future Directions | Cardiology in review (2026) | ☐ |
| 0 | PMID42234428 | HRS-7535 for Type 2 Diabetes Inadequately Controlled With Metformin: A Randomized Clinical Trial | JAMA network open (2026) | ☐ |
| 0 | PMID42251860 | Efficacy and safety of once-weekly cagrilintide-semaglutide (CagriSema) in adults with type 2 diabetes inadequately controlled on diet and exercise (REIMAGINE 1): a randomised, double-blind, placebo-controlled, phase 3a study | The lancet. Diabetes & endocrinology (2026) | ☐ |
| 0 | PMID42289243 | SGLT2 inhibitors and GLP-1 receptor agonists in atrial fibrillation treatment: Clinical evidence and mechanistic insights | Metabolism: clinical and experimental (2026) | ☐ |
| 0 | PMID42309434 | Adjunctive GLP1 Receptor Agonists in Patients With Inflammatory Bowel Diseases and Obesity and/or Diabetes: A Target Trial Emulation | Clinical gastroenterology and hepatology : the official clinical practice journal of the American Gastroenterological Association (2026) | ☐ |
| 0 | PMID42321509 | Adjunctive Dulaglutide in Relapsing-Remitting Multiple Sclerosis: A Randomized Open-Label Proof-of-Concept Trial | Neurology and therapy (2026) | ☐ |
| 0 | PMID42418075 | Incidence of Gastrointestinal Adverse Events and Healthcare Resource Utilization in Patients Using Glucagon-Like Peptide-1 Analogs: A Real-World Study | Digestive diseases and sciences (2026) | ☐ |
| 0 | PMID42496918 | Comparison of non-insulin glucose-lowering therapies on coronary atherosclerotic plaque progression in diabetes | La Radiologia medica (2026) | ☐ |
| 0 | PMID42712110 | Association of GLP-1 Receptor Agonists Versus Other Antidiabetic Medications With Cardiovascular, Renal, and Liver Outcomes in Patients With Type 2 Diabetes | Diabetes, obesity & metabolism (2026) | ☐ |
| 0 | PMID42786044 | Multiorgan protective therapy for CKM syndrome: the role of GLP-1-based therapies and SGLT2 inhibitors | European journal of internal medicine (2026) | ☐ |
