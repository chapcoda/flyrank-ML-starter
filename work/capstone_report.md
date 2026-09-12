# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Noah Chapman
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/chapcoda/flyrank-ML-starter
- **Date:** September 2026

## 0. Abstract

Can a model identify which published pages are about to lose search visibility, early
enough for a content team to act? Using 50,840 pages across 37 client portfolios from the
FlyRank internship warehouse — features from February–April 2026, outcomes from May 2026 —
I built a client-held-out Random Forest and compared it against the industry heuristic that
stale content decays. The heuristic did not merely underperform: it scored a ROC AUC of
0.298, below chance, meaning it ranked pages consistently wrong, and I traced that inversion
to a feature whose entire interquartile range collapsed to a single value. The model
improved on it (Precision@50 of 0.62 against 0.26; AUC 0.637 against 0.298), but auditing
that result surfaced three problems that matter more than the improvement: a one-directional
label contamination affecting 17.7% of positive cases, an evaluation metric that returned
five different values across five runs on a fixed seed, and a test set in which 87% of pages
belonged to a single client. The output is decision-support for sequencing a weekly content
review queue — not a count of declining pages, and not evidence that acting on them changes
the outcome.

## 1. Problem framing

**The decision.** An SEO agency has more pages than review hours. A team managing tens of
thousands of pages across dozens of clients cannot inspect them all, so the operative
question is never "which pages are declining" but "which twenty should I open this week."

**Unit of analysis:** the page (`content_hash_id`), within a client.

**Output:** a per-client ranked queue of ten pages, each with a plain-language reason code.

**The action a human takes:** an account manager opens the flagged pages in order and
decides whether to refresh, consolidate, re-target, or leave alone. The model orders
attention; the human makes every content decision.

**Cost of a wrong call.** A false positive costs a few hours of review on a page that was
fine — recoverable. A false negative means a page with existing visibility quietly decays
until recovery is expensive. The asymmetry favours a queue that surfaces marginal candidates
over one that misses real ones, which is part of why ranking beats classification here.

**Why ML helps at all.** The alternative is a hand-written rule, and the honest finding of
this project is that the obvious hand-written rule was worse than useless on this data
(Section 3). That does not make ML necessary — it makes the question empirical, which is the
point of building a baseline first.

## 2. Data safety

**Source.** `FlyRank/internship-warehouse`, a pseudonymised Parquet export on Hugging Face.
Tables used: `dim_content` (page metadata) and `fact_content_daily_performance` (daily
per-page search and analytics metrics, ~79M rows). DuckDB reads directly over `hf://`,
touching only the columns and monthly partitions each query needs — the full table is never
downloaded.

**Windows.** Features 2026-02-01 to 2026-04-30. Label denominator 2026-04-01 to 2026-04-30.
Outcome 2026-05-01 to 2026-05-31. No May data enters any feature, so the model is restricted
to what was knowable on 30 April.

**Deliberately excluded — label-derived.** `trend_direction`, `trend_pct`, and
`is_declining_label` were identified as label-derived in the Week 3 data contract and are
absent from the feature set. All 19 features were checked against that list plus the
intermediate label columns (`target_impressions`, `trailing_impressions`, `any_may_gsc`);
none are present.

**Deliberately excluded — inert.** `days_since_last_optimized` is 100% null. It was carried
in the first model version before detection; removing it changed Precision@50 and ROC AUC by
nothing, because `SimpleImputer` had been silently skipping the column throughout.

**Pseudonymous IDs are used for grouping only.** `client_hash_id` defines the train/test
split boundary and the per-client queue structure. It is never a feature. `content_hash_id`
is an identifier only.

**The availability trap, and how features were guarded.** The warehouse carries explicit
`gsc_data_available` and `ga4_data_available` flags because a client's data connection can
lapse, producing zeros that mean "not measured" rather than "measured as nothing." Every
feature aggregate is guarded by a `CASE WHEN gsc_data_available = TRUE` filter. Feature-window
GSC coverage is 100%, so the model reads observed data for every page it scores.

**Where the guard was not applied — and what it cost.** The same guard was not applied to the
*label*. Details in Section 5; the short version is that this is the project's most
consequential finding and it is a self-inflicted one.

**Public-safety confirmation.** No client names, page URLs, or search query text appear
anywhere in `work/`. All identifiers are hashed. Figures in the deployed paper relabel
clients as A–F.

## 3. Baseline

**The rule.** Rank pages by a staleness bucket weight multiplied by log-dampened impressions.
The bucket weights are derived from observed decline rates by age band (31–90 days weighted
1.00, 91–180 days 0.066, 181–365 days 0.008), and `log1p(impressions_90d)` dampens the
visibility term so a page with existing traffic outranks an invisible one at equal staleness.

**Why it is a fair comparison.** It encodes the standard industry heuristic — stale content
decays, refresh your oldest pages first — which is also the recommendation of FlyRank's own
published research. It is evaluated on exactly the same held-out rows, the same split, and
the same metric as the model. It is transparent, cheap, and is what a content team would
plausibly do without any model at all.

**Its numbers, on the same data and metric:**

| | Precision@50 | ROC AUC |
|---|---|---|
| Baseline | 0.260 | 0.298 |
| Model | 0.620 | 0.637 |
| Test base rate | 0.506 | — |

An AUC of 0.298 is the interesting number and Section 6 explains it.

One detail worth recording: the baseline returns exactly 0.260 and 0.298 on every run,
because it is arithmetic over fixed columns with no fitted model. The model's numbers move
(Section 5). The instability belongs to the model's probability estimates, not to the data or
the split.

## 4. Model / analysis

**Method.** `RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
class_weight='balanced', random_state=42, n_jobs=1)`, inside a pipeline with median
imputation for numeric features and one-hot encoding for categoricals, fitted on training
data only.

**Why it fits the lane.** Lane 2 needs a ranking over heterogeneous page-level features with
non-linear interactions and mixed types, trained on tens of thousands of rows on a free CPU.
A forest handles that without scaling or feature engineering, gives calibrated-enough
probabilities to rank by, and supports permutation importance for interpretation. It is not
chosen as the best possible model — it is chosen as a reasonable one, and no second model
class was tried, which Section 7 records as a limitation.

`n_jobs=1` is deliberate. See Section 5.

**The feature list — 19 features, all from Feb–April or static metadata.**

*Search performance:* `impressions_90d`, `clicks_90d`, `avg_position_weighted`, `ctr`
*Analytics:* `sessions_90d`, `engaged_sessions_90d`, `engagement_sec_90d`,
`sessions_organic_90d`, `engagement_rate`
*Content:* `char_count`, `word_count`, `content_age_days`, `days_since_last_update`
*Market:* `search_volume`, `competition`, `cpc`
*Categorical:* `content_type`, `main_intent`, `competition_level`

**Left out on purpose:** the three label-derived fields and `days_since_last_optimized`
(Section 2). `backlinks` was pulled in the query but not used.

**A feature-quality caveat.** `char_count` and `word_count` are null for 52.2% of the
modelling table and are median-imputed. No claim about content depth appears anywhere in this
work, and none should be drawn from it.

**Target definition, in one sentence.** A page is labelled *declining* if its May 2026
search impressions are at or below 75% of its April 2026 impressions, *stable* otherwise, and
*undefined* if it had zero April impressions.

**What the definition excludes.** 17,544 pages — 34.5% of the portfolio — have zero April
impressions, making a proportional decline undefined. They are dropped. This work is
therefore about pages that *had* visibility and lost it, and says nothing about pages that
never achieved any.

## 5. Evaluation

**The split: client-grouped holdout.** `GroupShuffleSplit` at 80/20 on `client_hash_id`, so
no client appears on both sides: 23 training clients (30,819 rows), 6 test clients (2,477
rows), zero overlap.

**Why grouped rather than random.** Pages within a client share site architecture, publishing
cadence, analytics connection quality, and demand seasonality. A random row-level split lets
the model recognise the client from pages it has already seen, which is information that never
exists for a new client in deployment. The cost is measurable:

| split | test clients | Precision@50 | ROC AUC |
|---|---|---|---|
| naive row-level | 28 | 0.84 | 0.727 |
| client-grouped | 6 | 0.62 | 0.637 |

Identical model, features, and seed. The naive split inflates Precision@50 by 35% and AUC by
14%. A paper reporting 0.84 would not be lying about the arithmetic — it would be measuring
something that never happens.

**Metrics.** Precision@50 is primary, because the decision is "which pages do I review this
week." ROC AUC is reported alongside and is the more trustworthy of the two here, for reasons
below. **Test base rate is 0.506** — a near coin flip by construction of the 0.75× threshold,
so any precision figure must be read against it.

**Error analysis.** Three things the metric table does not show.

*The top 50 is one client.* 49 of the model's top 50 predictions come from a single client
and one from a second; four of six test clients contribute nothing. This is partly the model
and substantially the test set's own composition — 2,152 of the 2,477 held-out pages (87%)
belong to that one client.

*A third of the apparent hits are unverifiable.* 31 of the top 50 are labelled declining, but
12 of those 31 have no honest May search-console coverage and were labelled declining by the
`COALESCE` mechanism below. Reported Precision@50 is 0.62; the verifiable floor, if every
unverifiable page were wrong, is 0.38.

*The model's blind spot.* Earlier error analysis found that false negatives — pages the model
scored as safe that declined anyway — had markedly worse average position and a quarter of the
impressions of true negatives. The model reads "already near the floor" as stability, which is
the opposite of what a content team would want.

**The label is contaminated, one-directionally.** The label query used
`COALESCE(may_impressions, 0)`, so a page absent from the May partition was treated as having
earned zero impressions. Zero always clears the 0.75× threshold. **Missing outcome data can
therefore only ever produce a "declining" label and never a "stable" one.**

| label | pages | with no honest May GSC data |
|---|---|---|
| declining | 16,711 | 2,964 (17.7%) |
| stable | 16,585 | 0 (0.0%) |

The effect is visible at client level too. Across the 22 clients with at least 30 labelled
pages, the correlation between a client's May GSC coverage and its apparent decline rate is
**−0.517**. The two clients with sub-50% coverage average an 89% apparent decline rate;
clients with functioning coverage cluster between 52% and 59%. A substantial part of what this
label records is whether a client's analytics connection was working.

Two honest qualifications: the relationship is driven mainly by those two extreme clients, and
22 clients is a small sample for a correlation. This is suggestive portfolio-level evidence for
a mechanism that is certain at row level.

**The fix is specified and not applied.** Restricting the label to pages with honest May
coverage would correct it. It is not applied anywhere in this report, because the
identification and measurement of the problem is this work's contribution and a corrected model
would be a different piece of work reported against different numbers. Restricting to clean
labels narrows the per-client decline range at the top (0.96 → 0.91) and widens it at the
bottom (0.38 → 0.27), so real between-client variation survives the correction.

**The primary metric is unstable.** Across five runs of an identical pipeline — same seed,
same 33,296 rows, same 6 test clients — Precision@50 measured 0.52, 0.58, 0.60, 0.62, and
0.66. ROC AUC over the same runs stayed within 0.632–0.639. Changing only `n_jobs` from -1 to
1, a parallelism setting with no mathematical effect, moved Precision@50 by two points and
left AUC within 0.001. The mechanism is that floating-point accumulation order varies with
thread count, perturbing borderline probabilities enough to reshuffle the rank-50 boundary.
All results here use `n_jobs=1` with the sklearn version recorded. **Any Precision@50 figure
in this report should be read to one significant figure.**

## 6. Interpretation

**The baseline inverted, and the mechanism is understood.** An AUC of 0.298 is not a weak
result but a backwards one — a rule with no information scores 0.5, and inverting this rule's
output would score roughly 0.70.

The bucket distribution in the held-out set explains it completely. 2,279 of 2,477 pages —
**92%** — fall in the 31–90 day age bucket, with 5 pages in 0–30 days and 193 in 91–180 days.
The staleness weight is therefore a constant for nine pages in ten, and the baseline score
collapses to `log(impressions)` for almost the entire portfolio, with a small group of
91–180 day pages pushed to the bottom by a 15× weight penalty regardless of visibility.

So the rule was not ranking by staleness at all. It was ranking by visibility. And visibility
runs the wrong way for this outcome: declining pages averaged 2,633 impressions against 2,861
for stable ones, and 4.5 clicks against 9.2. A rule that ranks high-visibility pages first is
ranking the *less* likely to decline first, which is exactly what an AUC below 0.5 records.

**Where the signal actually is.** Permutation importance on the held-out set, AUC-scored:

| feature | importance |
|---|---|
| avg_position_weighted | 0.0490 |
| ctr | 0.0310 |
| clicks_90d | 0.0147 |
| sessions_90d | 0.0042 |
| sessions_organic_90d | 0.0038 |
| search_volume | 0.0013 |
| **days_since_last_update** | **−0.0004** |

Two features carry nearly everything: where a page ranks, and whether people click it when
they see it. Everything else is close to noise.

**A negative result, carefully stated.** `days_since_last_update` — the feature the entire
baseline was built on — scores −0.0004. Shuffling it slightly *improved* AUC, which is what a
feature carrying no signal looks like.

The careless reading is "content freshness does not predict decline." That is not what this
shows. The feature's 25th, 50th, and 75th percentiles are **all exactly 64 days** — half the
portfolio sits on a single value, with variation only in the tails. A tree splits on
thresholds, and a feature whose entire interquartile range collapses to one number offers
almost nothing to split on. **This window could not test the freshness hypothesis**, because
the pages in it were nearly all updated at the same time. The hypothesis is untested here,
not disproved.

That also partly exonerates the Week 4 rule: it may have been testing a reasonable idea
against a window incapable of evaluating it.

**The surprise I could not fully explain.** Declining pages have a *better* average position
than stable ones (16.2 vs 20.8). Higher-ranking pages should not be more likely to decline.
The most plausible reading connects to the label contamination — pages pulled into the
declining class by missing May data carry whatever Feb–April position they had, including good
ones. Confirming that requires re-running under the corrected label, which this work does not
do. It is recorded as an open question rather than resolved.

**What compounds.** Three observations that appeared separately across this project resolve
into one mechanism: the feature had no usable variation, so the rule degenerated into a
visibility ranking, and visibility points the wrong way for this outcome.

## 7. Recommendation

**The output: a per-client ranked queue.** Top 10 pages per client, each with a
plain-language reason. Per-client rather than global, because the global top 50 draws 49 of 50
pages from one client and would hand four of six account managers an empty list.

**How an editor uses it tomorrow.** Open the ten flagged pages for your client in rank order.
For each, read the reason code, check it against the client's own Search Console, apply the
business context the model cannot have, and decide. Expect to dismiss several.

**Why pages surface** (52 queue entries):

| reason | pages |
|---|---|
| above-median impressions, bottom-quartile CTR | 19 |
| bottom-quartile CTR | 18 |
| high score, no single dominant driver | 7 |
| ranks deep for this client | 4 |
| ranks deep + bottom-quartile CTR | 3 |
| ranks deep + above-median impressions, low CTR | 1 |

All thresholds are relative to the page's own client. Forty-one of 52 entries involve weak
click-through; eight involve deep ranking. Two codes never fire — no queue page is stale
relative to its client, and none has zero impressions. **The queue is not surfacing neglected
pages. It is surfacing visible, under-converting ones** — a different population than the
staleness rule would have produced.

**Confidence: directional, per client unproven.** Aggregate queue hit rate is 63.5% against a
50.6% base rate. Per client: two beat base rate, two are flat, two fall below it. An earlier
run of the identical pipeline swung one client by 24 points. **Ten-page samples cannot
establish which clients the ranking helps.** The defensible claim is that ranking helps
somewhat on average, and this evidence cannot say for whom.

**Check before acting on any entry.** 27 of 52 queue pages have under 100 impressions in
ninety days, where the ranking is close to noise. 23 of 52 have no honest GA4 coverage, so
engagement signals are imputed. Verify the reason code against the client's console, and
supply the business context — a discontinued product, an ended campaign, a deliberate
deindexing — that the model has no access to.

One check came back clean: all 52 queue pages have complete search-console coverage in the
feature window. The contamination affects the *outcome* window, so it distorts how the model
was trained and scored, not what it reads when ranking a page today.

**Explicit limits.**

*Do not automate content changes from this.* The queue orders attention; humans decide.

*Do not report it as a count of at-risk pages.* Precision@50 of 0.62 has a verifiable floor of
0.38.

*Do not deindex or delete on a queue position.* The model's measured blind spot is exactly the
already-low-visibility pages that keep declining.

*Do not treat this as evidence that intervention works.* The model identifies association, not
mechanism. Testing whether refreshing a flagged page prevents its decline requires a randomised
comparison nobody has run here.

**The claim, stated plainly.** For a content team with limited review hours, working this
queue is a better use of the first ten hours than working alphabetically, by publication date,
or by the staleness heuristic — which this work shows would point the wrong way. That is a
modest claim and it is the one the evidence supports.

## 8. Reproducibility

**Re-run from a fresh clone.**

```
git clone https://github.com/chapcoda/flyrank-ML-starter
cd flyrank-ML-starter
pip install -q duckdb huggingface_hub scikit-learn pandas numpy matplotlib
```

Set a Hugging Face read token with access to `FlyRank/internship-warehouse` as the
environment variable `HF_TOKEN` (in Colab, as a Secret of that name — the notebooks read it
via `userdata.get("HF_TOKEN")` and never contain a token). Then open
`work/notebooks/capstone.ipynb` and run all cells in order. The notebook is self-contained:
it rebuilds the feature table, label, split, model, queue, and every figure from the
warehouse.

**Seeds and determinism.** `random_state=42` throughout — `GroupShuffleSplit`,
`RandomForestClassifier`, `train_test_split`, and `permutation_importance`. **`n_jobs=1` is
required for reproducibility**: with `n_jobs=-1`, floating-point accumulation order varies
with the thread count the runtime allocates, and Precision@50 moves by up to two points as a
result. This is documented in Section 5 rather than hidden.

**Environment.** Python 3.13, scikit-learn 1.6.1, DuckDB (latest at run time), pandas, numpy,
matplotlib. The sklearn version is printed by the setup cell and recorded in
`work/outputs/capstone_metrics.json`, because Precision@50 has varied across library versions.

**Holdout claim and how to check it.** The evaluation is a client-grouped holdout, not a
sealed one-shot evaluation — the split is rebuilt deterministically from `random_state=42` on
every run, and the model was refitted many times during development. I am not claiming a
blind single evaluation and the repo should not be read as supporting one.

What *is* checkable: the cell that builds the split and the held-out frame is the
`# MODEL, BASELINE, SPLIT, QUEUE` cell in `work/notebooks/capstone.ipynb`, and the metrics it
produced are committed at `work/outputs/capstone_metrics.json`. That file records both
metrics, both windows, the model specification, the sklearn version, row and client counts,
and the contamination figures — so every number in this report can be checked against a
committed artifact rather than taken on faith.

**Artifacts committed.**

| file | contents |
|---|---|
| `work/outputs/capstone_metrics.json` | every headline number, plus model spec and windows |
| `work/outputs/capstone_queue.csv` | the 52-page queue with anonymised client labels |
| `work/outputs/fig1_headline.png` | model vs baseline |
| `work/outputs/fig2_inversion.png` | age-bucket concentration and ROC curves |
| `work/outputs/fig3_contamination.png` | coverage vs apparent decline, 22 clients |
| `work/outputs/fig4_per_client.png` | queue hit rate vs base rate per client |

**The full track.** `work/notebooks/` holds the sequence this report is built from:
`w03_data_contract.ipynb` (grain and availability checks), `w04_baseline_score.ipynb` (the
baseline), `w05_model.ipynb` (the model), `w06_validation_audit.ipynb` (the audit that found
the contamination), `w07_action_playbook.ipynb` (the queue), and `capstone.ipynb`.

**Known deviation.** The label fix described in Section 5 is specified but not applied. Any
re-run reproduces the contaminated label, deliberately, so the reported numbers match.

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** — a pseudonymised export of production search
performance covering roughly 79 million daily page-level records across 37 client portfolios.

Data provided by [FlyRank](https://flyrank.ai).

All identifiers in this work are hashed. No client names, page URLs, or search query text
appear in the dataset or in this report.
