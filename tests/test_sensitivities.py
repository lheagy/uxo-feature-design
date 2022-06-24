import numpy as np

import discretize
from SimPEG import tests

from polarizability_model import SimulationPolarizabilityModel, Survey, MagneticControlledSource, MagneticFluxDensityReceiver


dx = 10
nc = 5
npad = 0
pf = 1.3
hx = [(dx, npad, -pf), (dx, nc), (dx, npad, pf)]
hy = [(dx, npad, -pf), (dx, nc+1), (dx, npad, pf)]
mesh = discretize.TensorMesh([hx, hy], origin="CC")
locations = np.zeros((mesh.nC, 3))
locations[:, :2] = mesh.gridCC

# survey
receivers_x = np.linspace(-5, 5, 20) #np.r_[0, 0.25]
receivers_y = np.r_[0]
receivers_z = np.r_[10]

receiver_locs = discretize.utils.ndgrid(receivers_x, receivers_y, receivers_z)
receivers = MagneticFluxDensityReceiver(locations=receiver_locs, components=["x", "y", "z"])

x_nodes = 1*np.r_[-1, 1, 1, -1, -1]
y_nodes = 1*np.r_[-1, -1, 1, 1, -1]
z_nodes = 0.25*np.ones_like(x_nodes)

src = MagneticControlledSource(
    receiver_list=[receivers], location=np.c_[x_nodes, y_nodes, z_nodes], current=100
)

offset = np.r_[0.5, 0.5, 0]
src2 = MagneticControlledSource(
    receiver_list=[receivers], location=np.c_[x_nodes + offset[0], y_nodes + offset[1], z_nodes + offset[2]], current=100
)



survey = Survey([src, src2])

# simulation
sim = SimulationPolarizabilityModel(locations, survey)

x0=np.random.rand(mesh.n_cells*3)*10
def deriv_test(x):
    return sim.dpred(x), lambda x: sim.Jvec(x0, x)

passed = tests.check_derivative(deriv_test, x0, dx=100*np.random.rand(len(x0)), num=4, plotIt=False)

if passed is False:
    raise Exception("Derivative test failed")

m = np.random.rand(mesh.nC*3)*10
v = np.random.rand(survey.nD)
w = np.random.rand(len(m))

vJw = v.dot(sim.Jvec(m, w))
wJtv = w.dot(sim.Jtvec(m, v))

print(f"adjoint test. vJw: {vJw:1.2e}, wJtv: {wJtv:1.2e}")
if np.linalg.norm(vJw - wJtv) > 0.5*(np.abs(vJw) + np.abs(wJtv)) * 1e-6:
    raise Exception("Adjoint test failed")
