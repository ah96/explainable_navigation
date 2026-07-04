# Explanation methods

All costmap methods share one **perturbation dataset**: the local costmap is
segmented into obstacles, each obstacle is toggled on/off, and the deterministic
planner is queried for the resulting local plan; the target is the plan's
**deviation** from the global plan (`nav_explain_core.trajectory.deviation`). The
methods differ in how they turn that dataset into an explanation.

## LIME (`nav_explain_core.lime`)
Fits a kernel-weighted Ridge surrogate over the perturbations. Each obstacle gets
an additive coefficient: **green** if its presence increases the deviation, **red**
if it decreases it. Fast (`one_off` mode = one perturbation per obstacle) and the
basis of the quality/robustness/runtime evaluation.

## SHAP (`nav_explain_core.shap`)
Computes **exact Shapley values** over the full `2^n` perturbation set (feasible
because segment counts are small). Each obstacle's value is its average marginal
contribution to the deviation across all obstacle subsets. The values satisfy
*efficiency* (they sum to `deviation(all) − deviation(none)`), giving a principled,
deterministic companion to LIME's approximate coefficients.

## Anchors (`nav_explain_core.anchors`)
Searches for a minimal set of obstacles whose presence **anchors** the decision:
`precision = P(robot deviates | these obstacles present) ≥ τ`. Because precision
is computed exactly over the enumerated perturbations, a deterministic beam search
replaces the original stochastic bandit. Reads as a rule: *"while these obstacles
are present, the robot deviates."*

## Counterfactual (`nav_explain_core.counterfactual`)
Finds the smallest set of obstacles whose **removal** drops the deviation below a
target — *"if these obstacles were gone, you wouldn't deviate."* The complement of
an anchor, and often the most actionable explanation.

## Visuo-textual (`nav_explain_core.semantic`)
Explains over a **semantic map** instead of raw superpixels: the interpretable
features are `(object, affordance)` pairs (movability, openability). Perturbing a
feature applies its affordance (hide a movable object, open a door). LIME then
attributes the deviation to each pair, colouring the semantic map; **Qualitative
Spatial Reasoning** places the responsible object relative to the robot
("left/front of me"), and templates produce descriptive + suggestive sentences
("Dear human, please open the door right/front of me").

## Qualitative Spatial Reasoning (`nav_explain_core.semantic.qsr`)
- **single-cross** intrinsic calculus: four quadrants of the robot's heading frame
  (`left/front`, `right/front`, `left/back`, `right/back`);
- **ternary** calculus: distance (`close`/`distant`) × direction
  (`front`/`left`/`back`/`right`), e.g. `close-front`.

## Trajectory metrics (`nav_explain_core.trajectory`)
`deviation` (the target) plus a reusable library: SSPD, Hausdorff, discrete
Fréchet, DTW, LCSS, EDR, ERP, OWD — vectorised with `scipy.spatial.distance.cdist`.
