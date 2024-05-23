'''
This script defines an example crocoddyl OCP with contact dynamics.
The movement features a (full) Talos humanoid, in contact with two feet, then with one single foot, doing some
reaching "taiichi" movements with a hand and a foot.

The example is taken from Crocoddyl taiichi example.
https://github.com/loco-3d/crocoddyl/blob/master/examples/humanoid_taichi.py
'''
# %jupyter_snippet import
import example_robot_data as robex
import numpy as np
import pinocchio
import crocoddyl
# %end_jupyter_snippet

# %jupyter_snippet loadrobot
# ### Load robot
robot = robex.load("talos")
robot_model = robot.model
# The robot data will be used at config time to define some values of the OCP
robot_data = robot_model.createData()
# %end_jupyter_snippet

# %jupyter_snippet hyperparameters
# ### Hyperparameters

# Set integration time
DT = 5e-2
T = 40

# Initialize reference state, target and reference CoM

hand_frameName = "gripper_left_joint"
rightFoot_frameName = "right_sole_link"
leftFoot_frameName = "left_sole_link"

# Main frame Ids
hand_id = robot_model.getFrameId(hand_frameName)
rightFoot_id = robot_model.getFrameId(rightFoot_frameName)
leftFoot_id = robot_model.getFrameId(leftFoot_frameName)

# Init state
q0 = robot_model.referenceConfigurations["half_sitting"]
x0 = np.concatenate([q0, np.zeros(robot_model.nv)])

# Reference quantities
pinocchio.framesForwardKinematics(robot_model, robot_data, q0)
comRef = (robot_data.oMf[rightFoot_id].translation + robot_data.oMf[leftFoot_id].translation) / 2
comRef[2] = pinocchio.centerOfMass(robot_model, robot_data, q0)[2].item()

in_world_M_foot_target_1 = pinocchio.SE3(np.eye(3), np.array([0.0, 0.4, 0.0]))
in_world_M_foot_target_2 =  pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35]))
in_world_M_hand_target = pinocchio.SE3(np.eye(3), np.array([0.4, 0, 1.2]))
# %end_jupyter_snippet

# %jupyter_snippet init_display
# ### DISPLAY
# Initialize viewer
from wan2024.meshcat_viewer_wrapper import MeshcatVisualizer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %end_jupyter_snippet

# ### CROCODDYL OCP
# ### CROCODDYL OCP
# ### CROCODDYL OCP
# %jupyter_snippet state
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)
# %end_jupyter_snippet

# %jupyter_snippet contacts
# ### Contact model
# Create two contact models used along the motion
supportContactModelLeft = crocoddyl.ContactModel6D(
    state,
    leftFoot_id,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state,
    rightFoot_id,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)

contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel1Foot.addContact(rightFoot_frameName + "_contact", supportContactModelRight)

contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet.addContact(leftFoot_frameName + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot_frameName + "_contact", supportContactModelRight)
# %end_jupyter_snippet

# %jupyter_snippet costs
# ### Cost model
# Cost for joint limits
maxfloat = 1e25
xlb = np.concatenate(
    [
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        robot_model.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        robot_model.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# Cost for state and control
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xTActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

# Cost for target reaching: hand and foot
handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state, hand_id, in_world_M_hand_target, actuation.nu
)
handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1] * 3 + [0.0001] * 3) ** 2
)
handTrackingCost = crocoddyl.CostModelResidual(
    state, handTrackingActivation, handTrackingResidual
)

# For the flying foot, we define two targets to successively reach
footTrackingResidual1 = crocoddyl.ResidualModelFramePlacement(
    state, leftFoot_id, in_world_M_foot_target_1, actuation.nu
)
footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1, 1, 0.1] + [1.0] * 3) ** 2
)
footTrackingCost1 = crocoddyl.CostModelResidual(
    state, footTrackingActivation, footTrackingResidual1
)
footTrackingResidual2 = crocoddyl.ResidualModelFramePlacement(
    state,
    leftFoot_id,
    in_world_M_foot_target_2,
    actuation.nu,
)
footTrackingCost2 = crocoddyl.CostModelResidual(
    state, footTrackingActivation, footTrackingResidual2
)

# Cost for CoM reference
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack = crocoddyl.CostModelResidual(state, comResidual)
# %end_jupyter_snippet

# ### Action models
# %jupyter_snippet actions
# Create cost model per each action model. We divide the motion in 3 phases plus its
# terminal model.

# Phase 1: two feet in contact, hand reach the target
runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, runningCostModel1
)
runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)

# Phase 2: only right foot in contact, hand stays on target, left foot to target 1
runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, runningCostModel2
)
runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)

# Phase 3: only right foot in contact, hand stays on target, left foot to target 2
runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, runningCostModel3
)
runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)

# Terminal cost: nothing specific
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, terminalCostModel
)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)
# %end_jupyter_snippet

# ### OCP Problem definition and solver
# %jupyter_snippet problem_and_solver
problem = crocoddyl.ShootingProblem(
    x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel
)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
solver.th_stop = 1e-7
solver.setCallbacks(
    [
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ]
)
# %end_jupyter_snippet

# %jupyter_snippet solve
# ### Warm start from quasistatic solutions
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 500, False, 1e-9)
# %end_jupyter_snippet

# ### Display and verbose
# Visualizing the solution in meshcat
# %jupyter_snippet display
viz.play([x[:robot.model.nq] for x in solver.xs],DT)
# %end_jupyter_snippet

# Get final state and end effector position
xT = solver.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[: state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
com = pinocchio.centerOfMass(robot_model, robot_data, xT[: state.nq])
finalPosEff = np.array(
    robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T.flat
)

print("Finally reached = ({:.3f}, {:.3f}, {:.3f})".format(*finalPosEff))
print(
    "Distance between hand and TARGET = {:.3E}".format(
        np.linalg.norm(finalPosEff - in_world_M_hand_target.translation)
    )
)
print(f"Distance to default state = {np.linalg.norm(x0 - np.array(xT.flat)):.3E}")
print(f"XY distance to CoM reference = {np.linalg.norm(com[:2] - comRef[:2]):.3E}")
