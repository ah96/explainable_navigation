# One explanation of a local navigation decision.
string    method          # lime | shap | anchors | counterfactual
int32[]   segment_ids      # interpretable feature (segment) ids
float32[] values           # per-segment importance (coefficient / Shapley value), parallel to segment_ids
float32   deviation        # local-plan deviation from the global plan
string    text             # human-readable explanation
