"""nav_explain_costmap: explain a robot's local costmap navigation decision.

Offers four explanation methods over the same scenario and shared perturbation
dataset: LIME, SHAP, Anchors and counterfactual. Depends only on
``nav_explain_core`` for the algorithms; adds scenarios, rendering, evaluation
and a live ROS 2 node.
"""

__version__ = "0.1.0"
