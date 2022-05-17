import numpy as np
import discretize
from discretize.utils import mkvc
from geoana.em.static import MagneticDipoleWholeSpace
from polarizability_model import SimulationPolarizabilityModel, Survey, MagneticUniformSource, MagneticFluxDensityReceiver


plot_it = False

# create a mesh and model
dx = 10
nc = 5
npad = 0
pf = 1.3
hx = [(dx, npad, -pf), (dx, nc), (dx, npad, pf)]
hy = [(dx, npad, -pf), (dx, nc+1), (dx, npad, pf)]
mesh = discretize.TensorMesh([hx, hy], origin="CC")
model = np.zeros(mesh.n_cells*3)
model[-1] = 1

locations = np.zeros((mesh.nC, 3))
locations[:, :2] = mesh.gridCC

# survey
receivers_x = np.linspace(-5, 5, 20) #np.r_[0, 0.25]
receivers_y = np.r_[0]
receivers_z = np.r_[10]

receiver_locs = discretize.utils.ndgrid(receivers_x, receivers_y, receivers_z)
receivers = MagneticFluxDensityReceiver(locations=receiver_locs, components=["x", "y", "z"])

src = MagneticUniformSource(receiver_list=[receivers], orientation="z", amplitude=1)

survey = Survey([src])

# create the simulation
sim = SimulationPolarizabilityModel(locations, survey)

# simulate data
dpred = sim.dpred(model)

# analytic solution
dipole = MagneticDipoleWholeSpace(location=locations[-1, :], orientation="z")
dana = dipole.magnetic_flux_density(receiver_locs)

if plot_it:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i, a in enumerate(ax):
        a.plot(receivers_x, dpred[i::3], "s", label="sim")
        a.plot(receivers_x, dana[:, i], "o", label="geoana")
    ax[0].legend()

if not np.allclose(dpred, mkvc(dana)):
    raise Exception("Solutions for the analytic and simulation don't match")
