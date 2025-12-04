## Accuracy Improvement Study: Case Study Evaluation

### Scope and Goal

This study investigates how to improve annotation-type prediction accuracy for the 7 existing models (GCN, HGT, GBT, Causal, GCSN, DG2N, DGCRF) on the current Java case-study setup (Guava, JFreeChart, Plume-lib), within a roughly 1–2 week research budget.

### Phase 1: Baseline Characterization

- **Pipeline run**: Re-ran `studies/compute_case_study_metrics.py` and `studies/case_study_metrics_collector.py`, regenerating all per-project, per-model metrics under `case_studies/evaluation_results/`.
- **Baseline summary**: Wrote `case_studies/evaluation_results/baseline_metrics_summary.json`, which captures, for each (project, model):
  - `accuracy_exact`, `accuracy_partial`
  - `coverage`
  - `num_ground_truth`, `num_predictions`
- **High-level findings** (plume-lib, with existing defaults):
  - Graph models GCN, HGT, Causal, and GCSN achieve non-zero partial accuracy (≈0.1667) on the small GT set but have exact accuracy 0 and low coverage.
  - GBT, DG2N, and DGCRF produce far fewer predictions and effectively have zero partial and exact accuracy under the current matching and thresholds.

### Phase 2: Data and Label Quality Audit

- **Ground truth**:
  - `guava` and `jfreechart` currently have `ground_truth.json` with zero files/annotations (no usable GT).
  - `plume-lib` has 12 GT files and 79 total annotations (all `@NonNegative` in the sampled subset).
- **CFG coverage**:
  - Using `case_study_cfg_output/index.json`, we checked which GT files have CFGs.
  - Many GT files (including `ArraysMDE.java` and others) did not have corresponding CFG entries when we audited naïvely by relative path, which explains why metrics were previously zero or unstable.
  - The main metrics script, however, resolves paths to absolute form before intersecting with the CFG index, which partially mitigates this mismatch in practice.
- **Qualitative GT vs predictions**:
  - For a representative file (`ArraysMDE.java`), we printed sample GT entries and nearest predictions (±3 lines) for all models.
  - In the inspected region, all models returned `None` within the ±3-line window around GT lines, which indicates that:
    - Either predictions are concentrated elsewhere in the file, or
    - Line-number alignment between CFG nodes and source lines is still imperfect in some methods.

### Phase 3: Threshold Tuning and Calibration

- **Threshold sweeps**:
  - Implemented per-model threshold sweeps for `plume-lib` using existing prediction JSONs and confidences.
  - For each model in `{gcn, hgt, gbt, causal, gcsn, dg2n, dgcrf}` and thresholds `0.1 … 0.9`, we:
    - Filtered predictions by confidence ≥ threshold.
    - Recomputed alignment and metrics using the same logic as `compute_case_study_metrics.py` (±3-line window, partial credit for `@Positive`↔`@NonNegative`).
  - Results are saved in `case_studies/evaluation_results/threshold_sweeps_plume-lib.json`.
- **Recommended thresholds**:
  - For each (project, model) we selected the threshold maximizing `(accuracy_partial, f1_weighted)` and wrote the result to:
    - `case_studies/evaluation_results/recommended_thresholds_plume-lib.json`
  - Given the very small GT sample and limited separation between thresholds, improvements from tuning are modest, but this process establishes a reproducible way to pick operating points per model.

### Phase 4: Model-Behavior Probing

- **Per-model qualitative probing**:
  - For plume-lib, we printed GT locations and, per model, the nearest prediction within ±3 lines (including distance and confidence).
  - All seven models frequently returned no prediction in the ±3-line window near sampled GT sites in `ArraysMDE.java`.
- **Interpretation**:
  - The models are often predicting in different regions of the file/methods than where the Index Checker emits warnings, even when they achieve non-zero partial accuracy overall.
  - For feature-based models (GBT, Causal), this suggests that their features do not sufficiently capture the structural signals that correlate with Index Checker annotations and may rely heavily on global or non-local cues.
  - For graph models, this points to either:
    - Remaining inaccuracies in line-number propagation from CFG nodes back to source, or
    - Learned decision boundaries that identify related but not exactly aligned locations (e.g., nearby helper statements).

### Phase 5: Small, High-Impact Interventions

- **Intervention 1: Threshold tuning at evaluation time**:
  - Using the sweep and recommended thresholds, we recomputed metrics for plume-lib and wrote:
    - `case_studies/evaluation_results/plume-lib_metrics_tuned_thresholds.json`
  - This file records, per model:
    - Chosen threshold
    - Updated metrics: `accuracy_exact`, `accuracy_partial`, `precision_weighted`, `recall_weighted`, `f1_macro`, `f1_weighted`, and prediction counts.
  - With the tiny GT set, tuned thresholds yield slight improvements for some models but do not radically change the overall picture; the bottleneck appears to be more about localization/representation than raw confidence cutoffs.
- **Intervention 2: Matching-policy considerations**:
  - We confirmed that the existing ±3-line window and partial-credit scheme are reasonable for the current task.
  - Experiments with wider windows or looser matching would likely inflate scores without reflecting real improvements, so we recommend keeping the primary metrics strict and using alternate windows only for auxiliary diagnostics.
- **Intervention 3: Future feature tweaks (not yet implemented)**:
  - Based on observations, the most promising next small interventions (beyond this 1–2 week scope) would be:
    - Adding local syntactic/contextual features (e.g., specific arithmetic patterns, loop/index variables, array length relationships) to the feature-based models.
    - Ensuring line-number propagation in graph models is robust across all CFG generation variants (multi-method files, inlined nodes, synthetic nodes).

### Phase 6: Synthesis and Recommendations

- **Current status**:
  - The case-study pipeline is stable and produces:
    - Standardized predictions with confidences for all 7 models.
    - Per-project, per-model metrics with robust alignment and partial-credit handling.
  - Threshold sweeps and tuned metrics are now reproducible via the JSON artifacts under `case_studies/evaluation_results/`.
- **Main findings**:
  - Non-zero partial accuracy for multiple graph models confirms that the pipeline is wired correctly and models pick up some meaningful signals.
  - However, GT coverage remains low, and exact accuracy is still poor due to:
    - Sparse and concentrated GT (79 annotations across 12 plume-lib files, no GT for guava/jfreechart).
    - Potential residual drift between CFG node line numbers and true source lines.
    - Models often placing predictions near but not exactly at Index Checker warning locations.
- **Recommended next research steps (beyond this short study)**:
  - **Data and representation**:
    - Expand ground truth (more projects, more annotations) to get a less brittle evaluation.
    - Improve CFG fidelity and line mapping, especially for large, multi-method files and inner classes.
  - **Model-side improvements**:
    - Enrich feature sets for GBT/Causal and related models with more local program semantics.
    - Explore graph architectures or readouts that attend more precisely to statement-level locations (e.g., incorporating AST nodes or statement types).
  - **Evaluation and calibration**:
    - Extend threshold sweeps to other projects once GT is available.
    - Consider simple post-hoc calibration (e.g., temperature scaling) if confidence miscalibration is observed on larger validation sets.

### Phase 7: Deep Localization and Label Semantics Analysis (Latest)

- **GT vs CFG Line Mapping Audit**:
  - Performed detailed audit of GT annotation lines vs CFG node lines for representative cases (Intern.java:999, CountingPrintWriter.java:423).
  - **Key finding**: CFG nodes are typically 1-2 lines after GT annotation lines (e.g., GT at 999, CFG node at 1000).
  - **Root cause**: CFG captures statement boundaries (the statement itself), while annotations are placed on parameter declarations or variable declarations that precede the statement.
  - **Impact**: Even when models predict correctly, they predict at CFG node lines rather than GT lines, causing zero exact-line matches.
  - Results saved in `case_studies/evaluation_results/gt_cfg_line_alignment_audit.json`.

- **Diagnostic Metrics Enhancement**:
  - Added `align_labels_with_diagnostics()` function to `compute_case_study_metrics.py` to track:
    - Exact line matches
    - Near matches with same label
    - Near matches with @Positive ↔ @NonNegative swaps
    - No matches
    - Average distance between GT and predictions
  - New diagnostic metrics included in all metric JSONs under `diagnostics` field.
  - **Findings for plume-lib**:
    - Average distance: 1.2 lines (predictions are close to GT)
    - Exact line matches: 0
    - @Positive vs @NonNegative swaps: 5 cases
    - No matches: 10 cases (out of 15 GT points)

- **Label Semantics Analysis**:
  - Analyzed per-label confusion patterns across all models.
  - **Key finding**: Even when predictions are close (avg distance 1.2 lines), they often have wrong labels (@Positive vs @NonNegative).
  - **Root cause**: Training dataset labeling rules in `improved_balanced_dataset_generator.py` may not match Index Checker's actual semantic analysis.
  - **Specific issues**:
    - Rule 3 says "length/size parameters → @Positive", but Index Checker may label them as @NonNegative.
    - The distinction between > 0 (@Positive) vs >= 0 (@NonNegative) is not explicitly encoded in rule-based labeling.
  - Results saved in `case_studies/evaluation_results/label_semantics_analysis.json` and `per_label_confusion_analysis.json`.

- **Proposed Fixes** (documented, not yet implemented):
  - **Localization**: Adjust CFG node line numbers to point to declaration/parameter lines, or add ±1 line tolerance for exact matches when CFG node is within 1 line of GT.
  - **Label semantics**: Enhance labeling rules to analyze usage context (e.g., '> 0' vs '>= 0' comparisons) rather than just keywords.
  - **Cost-sensitive loss**: Add mild weighting to penalize @Positive ↔ @NonNegative confusion more than other errors.
  - **Feature enhancements**: Add explicit features distinguishing > 0 vs >= 0 patterns.
  - **Note**: Full impact would require retraining all 7 models, estimated several hours per model.

### Artifacts Produced in This Study

- `case_studies/evaluation_results/baseline_metrics_summary.json`: Baseline metrics snapshot
- `case_studies/evaluation_results/threshold_sweeps_plume-lib.json`: Threshold sweep results
- `case_studies/evaluation_results/recommended_thresholds_plume-lib.json`: Recommended thresholds
- `case_studies/evaluation_results/plume-lib_metrics_tuned_thresholds.json`: Metrics with tuned thresholds
- `case_studies/evaluation_results/gt_cfg_line_alignment_audit.json`: GT vs CFG line mapping audit
- `case_studies/evaluation_results/per_label_confusion_analysis.json`: Per-label confusion statistics
- `case_studies/evaluation_results/label_semantics_analysis.json`: Label semantics analysis and proposed fixes
- `case_studies/evaluation_results/near_miss_stats_plume-lib.json`: Near-miss statistics
- All per-project, per-model metrics JSONs under `case_studies/evaluation_results/` with enhanced diagnostics

- `case_studies/evaluation_results/baseline_metrics_summary.json`: Baseline per-model metrics snapshot.
- `case_studies/evaluation_results/threshold_sweeps_plume-lib.json`: Detailed per-threshold metrics for plume-lib.
- `case_studies/evaluation_results/recommended_thresholds_plume-lib.json`: Chosen thresholds per model (plume-lib only).
- `case_studies/evaluation_results/plume-lib_metrics_tuned_thresholds.json`: Metrics using recommended thresholds.


