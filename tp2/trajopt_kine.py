"""
Implement and solve the following nonlinear program:
decide q_0 ... q_T \in R^NQxT
minimizing   sum_t || q_t - q_t+1 ||**2 + || log( M(q_T)^-1 M^* ||^2 
so that q_0 = robot.q0
with M(q) \in SE(3) the placement of the robot end-effector, and M^* the target.

The following tools are used:
- the ur10 model (loaded by example-robot-data)
- pinocchio.casadi for writing the problem and computing its derivatives
- the IpOpt solver wrapped in casadi
- the meshcat viewer
"""

# %jupyter_snippet imports
import time
import casadi
import example_robot_data as robex
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from wan2024.meshcat_viewer_wrapper import MeshcatVisualizer
# %end_jupyter_snippet


### HYPER PARAMETERS
# %jupyter_snippet configurations
in_world_M_target = pin.SE3(pin.utils.rotate("y", 3), np.array([-0.5, 0.1, 0.2]))  # x,y,z
q0 = np.array([0, -3.14 / 2, 0, 0, 0, 0])
tool_frameName = "tool0"
# %end_jupyter_snippet

# %jupyter_snippet hyper
T = 10
w_run = 0.1
w_term = 1
# %end_jupyter_snippet

# --- Load robot model
robot = robex.load("ur10")
# %jupyter_snippet modeldata
robot.q0 = q0

# The pinocchio model is what we are really interested by.
model = robot.model
data = model.createData()
tool_id = model.getFrameId(tool_frameName)
# %end_jupyter_snippet

# --- Add box to represent target
# %jupyter_snippet viewer
# Open the viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# Add a vizualization for the target
boxID = "world/box"
viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
# Add a vizualisation for the tip of the arm.
tipID = "world/blue"
viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])

def displayScene(q, dt=1e-1):
    """
    Given the robot configuration, display:
    - the robot
    - a box representing tool_id
    - a box representing in_world_M_target
    """
    pin.framesForwardKinematics(model, data, q)
    M = data.oMf[tool_id]
    viz.applyConfiguration(boxID, in_world_M_target)
    viz.applyConfiguration(tipID, M)
    viz.display(q)
    time.sleep(dt)
# %end_jupyter_snippet

# %jupyter_snippet disptraj
def displayTraj(qs, dt=1e-2):
    for q in qs[1:]:
        displayScene(q, dt=dt)
# %end_jupyter_snippet

# %jupyter_snippet casadi
# --- Casadi helpers
cmodel = cpin.Model(model)
cdata = cmodel.createData()

cq = casadi.SX.sym("x", model.nq, 1)
cpin.framesForwardKinematics(cmodel, cdata, cq)

error3_tool = casadi.Function(
    "etool3", [cq], [cdata.oMf[tool_id].translation - in_world_M_target.translation]
)
error6_tool = casadi.Function(
    "etool6",
    [cq],
    [cpin.log6(cdata.oMf[tool_id].inverse() * cpin.SE3(in_world_M_target)).vector],
)
error_tool = error3_tool
# %end_jupyter_snippet

### PROBLEM
# %jupyter_snippet casadi_opti
opti = casadi.Opti()
var_qs = [opti.variable(model.nq) for t in range(T + 1)]
totalcost = 0
# %end_jupyter_snippet

# %jupyter_snippet casadi_q0
opti.subject_to(var_qs[0] == robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet casadi_runcost
for t in range(T):
    totalcost += w_run * casadi.sumsqr(var_qs[t] - var_qs[t + 1])
# %end_jupyter_snippet

# %jupyter_snippet casadi_termcost
totalcost += w_term * casadi.sumsqr(error_tool(var_qs[T]))
# %end_jupyter_snippet

### SOLVE
# %jupyter_snippet solve
opti.minimize(totalcost)
opti.solver("ipopt")  # set numerical backend
opti.callback(lambda i: displayScene(opti.debug.value(var_qs[-1])))

# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    sol_qs = [opti.value(var_q) for var_q in var_qs]
except:
    print("ERROR in convergence, plotting debug info.")
    sol_qs = [opti.debug.value(var_q) for var_q in var_qs]
# %end_jupyter_snippet

print("***** Display the resulting trajectory ...")
displayScene(robot.q0, 1)
displayTraj(sol_qs,1e-1)
