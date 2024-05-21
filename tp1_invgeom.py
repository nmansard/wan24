"""
Stand-alone inverse geometry for a manipulator robot with a 6d objective.

Implement and solve the following nonlinear program:
decide q \in R^NQ
minimizing   || log( M(q)^-1 M^* ||^2
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the
target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer
"""

# %jupyter_snippet imports
import time
import unittest
import example_robot_data as robex
import numpy as np
import casadi
import pinocchio as pin
import pinocchio.casadi as cpin
from wan2024.meshcat_viewer_wrapper import MeshcatVisualizer, colors
# %end_jupyter_snippet

# --- ROBOT AND VIZUALIZER

# %jupyter_snippet robot
robot = robex.load("ur10")
model = robot.model
data = robot.data
# %end_jupyter_snippet

# %jupyter_snippet visualizer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet task_params
robot.q0 = np.array([0, -np.pi / 2, 0, 0, 0, 0])

tool_id = model.getFrameId("tool0")

in_world_M_target = pin.SE3(
    pin.utils.rotate("x", np.pi / 4),
    np.array([-0.5, 0.1, 0.2]),
)
# %end_jupyter_snippet

print("Let's go to pdes ... with casadi")

# %jupyter_snippet visualizer_callback
# --- Add box to represent target
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
# %jupyter_snippet 1
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])

def displayProblem(q):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    in_world_M_tool = data.oMf[tool_id]
    viz.applyConfiguration(boxID, in_world_M_target)
    viz.applyConfiguration(tipID, in_world_M_tool)
    viz.display(q)
    time.sleep(1e-1)
# %end_jupyter_snippet visualizer_callback

# --- CASADI MODEL AND HELPERS

# %jupyter_snippet casadi_model
# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()
# %end_jupyter_snippet

# %jupyter_snippet cq
cq = casadi.SX.sym("q", model.nq, 1)
# %end_jupyter_snippet

# %jupyter_snippet casadi_fk
cpin.framesForwardKinematics(cmodel, cdata, cq)
# %end_jupyter_snippet

# %jupyter_snippet casadi_error
error_tool = casadi.Function(
    "etool",
    [cq],
    [
        cpin.log6(
            cdata.oMf[tool_id].inverse() * cpin.SE3(in_world_M_target)
        ).vector
    ],
)
# %end_jupyter_snippet

# --- OPTIMIZATION PROBLEM

# %jupyter_snippet casadi_computation_graph
opti = casadi.Opti()
var_q = opti.variable(model.nq)
totalcost = casadi.sumsqr(error_tool(var_q))
# %end_jupyter_snippet

# %jupyter_snippet ipopt
opti.minimize(totalcost)
opti.solver("ipopt")  # select the backend solver
opti.callback(lambda i: displayProblem(opti.debug.value(var_q)))

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_q = opti.value(var_q)
except:
    print("ERROR in convergence, plotting debug info.")
    sol_q = opti.debug.value(var_q)
# %end_jupyter_snippet

# %jupyter_snippet check_final_placement
print(
    "The robot finally reached effector placement at\n",
    robot.placement(sol_q, 6),
)
# %end_jupyter_snippet
