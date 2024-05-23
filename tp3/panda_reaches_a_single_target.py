'''
In this example test, we will solve the reaching-goal task with the Panda arm.
For that, we use the forward dynamics (with its analytical derivatives) based on Pinocchio
and featured in the Pinocchio-based front-end implementation of Crocoddyl. The main code is
inside the DifferentialActionModelFullyActuated class.
We set up 3 main costs: state regularization, control regularization wrt gravity, and target reaching.
For temporal integration, we use an Euler implicit integration scheme.

This example is based on https://github.com/Gepetto/supaero2023/blob/main/tp5/arm_example.py
'''

# %jupyter_snippet import
import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data as robex
import matplotlib.pylab as plt
# %end_jupyter_snippet

# %jupyter_snippet robexload
# First, let's load the Pinocchio model for the Panda arm.
robot = robex.load('panda')
# The 2 last joints are for the fingers, not important in arm motion, freeze them
robot.model,[robot.visual_model,robot.collision_model] = \
    pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],[8,9],robot.q0)
robot.q0 = robot.q0[:7].copy()
# %end_jupyter_snippet

# Hyperparameters of the movement
# %jupyter_snippet hyperparameters
HORIZON_LENGTH = 100
TIME_STEP = 1e-2
FRAME_TIP = robot.model.getFrameId("panda_hand_tcp")
GOAL_POSITION = np.array([.2,0.6,.5])
GOAL_PLACEMENT = pin.SE3(pin.utils.rpyToMatrix(-np.pi,-1.5,1.5), GOAL_POSITION)
REACH_DIMENSION = "3d" # "6d"
# %end_jupyter_snippet

# Configure viewer to vizualize the robot and a green box to feature the goal placement.
# %jupyter_snippet viz
from wan2024.meshcat_viewer_wrapper import MeshcatVisualizer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
viz.addBox('world/goal',[.1,.1,.1],[0,1,0,1])
viz.applyConfiguration('world/goal',GOAL_PLACEMENT)
# %end_jupyter_snippet

# %jupyter_snippet robot_model
# Set robot model
robot_model = robot.model
robot_model.armature = np.ones(robot.model.nv)*2 # Arbitrary value representing the true armature
robot_model.q0 = robot.q0.copy()
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])
# %end_jupyter_snippet

# Define the state to be x=(q,v) position and velocity of the robot in the configuration space.
# %jupyter_snippet state
state = crocoddyl.StateMultibody(robot_model)
# %end_jupyter_snippet

# Define the cost to be the sum of 3 terms: state regularisation xReg, control regularization uReg,
# and end-effector reaching. We create a special value for the terminal cost.
# %jupyter_snippet sumofcosts
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# %end_jupyter_snippet

# You can ask the end effector to reach a position (3d) or a placement (6d)
# %jupyter_snippet cost_goal
if REACH_DIMENSION == "3d":
    # Cost for 3d tracking || p(q) - pref ||**2
    goalTrackingRes = crocoddyl.ResidualModelFrameTranslation(state,FRAME_TIP,GOAL_POSITION)
    goalTrackingWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1]))
elif REACH_DIMENSION == "6d":
    # Cost for 6d tracking  || log( M(q)^-1 Mref ) ||**2
    goalTrackingRes = crocoddyl.ResidualModelFramePlacement(state,FRAME_TIP,GOAL_PLACEMENT)
    goalTrackingWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1, 1,1,1]))
else:
    assert( REACH_DIMENSION=="3d" or REACH_DIMENSION=="6d" )
goalTrackingCost = crocoddyl.CostModelResidual(state,goalTrackingWeights,goalTrackingRes)
runningCostModel.addCost("gripperPose", goalTrackingCost, .001)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4)
# %end_jupyter_snippet
    
# %jupyter_snippet cost_xreg
# Cost for state regularization || x - x* ||**2
# We set up different values for the integral cost and terminal cost term.

# Regularization is stronger on position than velocity (to account for typical unit scale)
xRegWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1, .1,.1,.1,.1,.1,.1,.1]))
xRegRes = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCost = crocoddyl.CostModelResidual(state,xRegWeights,xRegRes)
runningCostModel.addCost("xReg", xRegCost, 1e-2)

# Terminal cost for state regularization || x - x* ||**2
# Require more strictly a small velocity at task end (but we don't car for the position)
xRegWeightsT=crocoddyl.ActivationModelWeightedQuad(np.array([.5,.5,.5,.5,.5,.5,.5,  5.,5.,5.,5.,5.,5.,5.]))
xRegResT = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCostT = crocoddyl.CostModelResidual(state,xRegWeightsT,xRegResT)
terminalCostModel.addCost("xReg", xRegCostT, .1)
# %end_jupyter_snippet

# %jupyter_snippet cost_ureg
# Cost for control regularization || u - g(q) ||**2
uRegRes = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state,uRegRes)
runningCostModel.addCost("uReg", uRegCost, 1e-5)
# %end_jupyter_snippet

# %jupyter_snippet iam
# Next, we need to create the running and terminal action model.
# The forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.

# The actuation model is here trivial: tau_q = u.
actuationModel = crocoddyl.ActuationModelFull(state)
# Running model composing the costs, the differential equations of motion and the integrator.
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, runningCostModel), TIME_STEP)
runningModel.differential.armature = robot_model.armature
# Terminal model following the same logic, although the integration is here trivial.
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)
terminalModel.differential.armature = robot_model.armature
# %end_jupyter_snippet

# Define the optimal control problem.
# For this optimal control problem, we define HORIZON_LENGTH knots (or running action
# models) plus a terminal knot
# %jupyter_snippet shoot
problem = crocoddyl.ShootingProblem(robot_model.x0, [runningModel] * HORIZON_LENGTH, terminalModel)
# %end_jupyter_snippet

# %jupyter_snippet solve
# Solving it using DDP
# Create the DDP solver for this OC problem, verbose traces, with a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose(),
])

# Solving it with the DDP algorithm
ddp.solve([],[],1000)  # xs_init,us_init,maxiter
# %end_jupyter_snippet
#assert( ddp.stop == 1.9384159634520916e-10 )

'''
# %jupyter_snippet plot
# Plotting the solution and the DDP convergence
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(
    log.costs,
    log.pregs,
    log.dregs,
    log.grads,
    log.stops,
    log.steps,
    figIndex=2,
    show=False,
)
# %end_jupyter_snippet
print('Type plt.show() to display the plots')
'''

# %jupyter_snippet animate
# # Visualizing the solution in gepetto-viewer
viz.play([x[:robot.model.nq] for x in ddp.xs],TIME_STEP)
# %end_jupyter_snippet


from tp5.plot_ricatti import plotRicattiGains
#plotRicattiGains(ddp)
from mpl_toolkits.mplot3d import Axes3D

M = np.array([abs(K.T) for K in ddp.K])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z = np.meshgrid(*[range(n) for n in M.shape])
ax.scatter(x,y,z, c=M.flat,vmin=0,vmax=200,s=200,alpha=[min(1,v) for v in (M/200).flat])
ax.set_xlabel(f'Time (0..{len(ddp.K)})')
ax.set_ylabel(f'X [0..{len(ddp.xs[0])}[')
ax.set_zlabel(f'U [0..{len(ddp.us[0])}[')


fig,ax = plt.subplots(5,5)
x2,y2=np.meshgrid(range(len(ddp.xs[0])),range(len(ddp.us[0])))
for i in range(5):
    for j in range(5):
        ax[i,j].scatter(x2,y2,c=abs(ddp.K[(j*4+i)*4]),s=200,vmin=0,vmax=200)
        ax[i,j].set_ylabel(f'T={4*(j*4+i)}')

fig,ax = plt.subplots(5,5)
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(abs(ddp.K[4*(j*4+i)]),vmin=0,vmax=200)
        ax[i,j].set_ylabel(f'T={4*(j*4+i)}')
plt.show()


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
import unittest
class LocalTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue( len(ddp.xs) == len(ddp.us)+1 )
        self.assertTrue( np.allclose(ddp.xs[0],ddp.problem.x0) )
        self.assertTrue( ddp.stop<1e-6 )
        
if __name__ == "__main__":
    LocalTest().test_logs()
