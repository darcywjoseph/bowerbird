"""The benchmark: curate a dataset, train a model on it, and test the model.

Needs the ``bench`` extra, which installs the simulator and training libraries.
"""

import os

# The benchmark never draws a picture. On a machine with no display the
# simulator's other graphics settings fail when it is imported, and this one
# does not.
os.environ.setdefault("MUJOCO_GL", "glfw")
