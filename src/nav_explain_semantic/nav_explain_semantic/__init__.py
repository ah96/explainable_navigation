"""nav_explain_semantic — visuo-textual explanations of navigational choices.

Combines LIME over a semantic map (object-affordance features) with Qualitative
Spatial Reasoning and an ontology to produce simultaneous visual (coloured
per-object explanation maps) and textual explanations. Algorithms live in
``nav_explain_core.semantic``; this package adds scenarios, rendering and an
optional ROS2 node.
"""

__version__ = "0.1.0"
