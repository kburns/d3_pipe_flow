

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lz = 2 * np.pi
Nz = 256
Nphi = 256
Nr = 66
Re = 5300
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK443
safety = 0.25
max_timestep = safety * Lz / Nz
dtype = np.float64
mesh = None

# Bases
z_coord = d3.Coordinate('z')
disk_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = d3.DirectProduct(z_coord, disk_coords)
dist = d3.Distributor(cyl_coords, dtype=dtype, mesh=mesh)
z_basis = d3.Fourier(z_coord, size=Nz, bounds=(0, Lz), dealias=dealias, dtype=dtype)
disk_basis = d3.DiskBasis(disk_coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
z, phi, r = dist.local_grids(z_basis, disk_basis)

# Fields
u0 = dist.VectorField(cyl_coords, name='u0', bases=disk_basis.radial_basis)
u = dist.VectorField(cyl_coords, name='u', bases=(z_basis, disk_basis))
p = dist.Field(name='p', bases=(z_basis, disk_basis))
tau_u = dist.VectorField(cyl_coords, name='tau_u', bases=(z_basis, disk_basis.edge))
tau_p = dist.Field(name='tau_p')

# Substitutions
ez = dist.VectorField(cyl_coords, bases=disk_basis)
ez['g'][0] = 1
lift_basis = disk_basis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Problem
problem = d3.IVP([u, p, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - (1/Re)*lap(u) + lift(tau_u) = - u@grad(u) + (4/Re)*ez")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
u['g'][0] += 1 - r**2

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=5, max_writes=50)
snapshots.add_tasks(solver.state)
slices = solver.evaluator.add_file_handler('slices', sim_dt=0.25, max_writes=50)
slices.add_task(u(z=0), name='u(z=0)')
slices.add_task(u(phi=0), name='u(phi=0)')
slices.add_task(u(phi=np.pi), name='u(phi=pi)')
slices.add_task(u(r=0.95), name='u(r=0.95)')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@u)/2, name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, max(KE)=%f' %(solver.iteration, solver.sim_time, timestep, flow.max('KE')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

