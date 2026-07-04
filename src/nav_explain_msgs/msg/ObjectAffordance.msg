# One (object, affordance) contribution in a visuo-textual explanation.
int32   object_id
string  label              # object label, e.g. "cabinet"
string  affordance         # movability | openability
float32 coefficient        # LIME coefficient for this (object, affordance) pair
string  qsr                # qualitative spatial relation, e.g. "left/front"
