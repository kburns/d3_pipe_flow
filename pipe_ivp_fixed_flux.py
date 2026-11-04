"""
Pipe flow initial value problem at fixed flux in Dedalus.
With radius R and centerline velocity U, the Reynolds number is Re = R*U/nu.
This script is nondimensionalized such that R=1 and U=1, where U is the
centerline velocity of the laminar profile u = (1 - r**2)*ez.
The volume flux is held at its laminar value by a uniform axial forcing.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lz = 2 * np.pi
Nz = 128
Nphi = 192
Nr = 64
Re = 5300
dealias = 3/2
dtype = np.float64
stop_sim_time = 100
timestepper = d3.RK443
safety = 0.5
max_timestep = safety * Lz / Nz
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
u = dist.VectorField(cyl_coords, name='u', bases=(z_basis, disk_basis))
p = dist.Field(name='p', bases=(z_basis, disk_basis))
f = dist.Field(name='f')
tau_u = dist.VectorField(cyl_coords, name='tau_u', bases=(z_basis, disk_basis.edge))
tau_p = dist.Field(name='tau_p')

# Base flow
u0 = dist.VectorField(cyl_coords, name='u0', bases=(z_basis, disk_basis))
u0['g'][0] = 1 - r**2

# Substitutions
ez = dist.VectorField(cyl_coords, bases=disk_basis.radial_basis)
ez['g'][0] = 1
er = dist.VectorField(cyl_coords, bases=disk_basis.edge)
er['g'][2] = 1
lift_basis = disk_basis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
u1 = u - u0
tau_wall = - er @ d3.grad(ez@u)(r=1) / Re  # d3.radial picks the wrong component on DirectProduct coords

# Problem
problem = d3.IVP([u, p, tau_u, tau_p, f], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - (1/Re)*lap(u) + lift(tau_u) - f*ez = - u@grad(u)")
problem.add_equation("integ(ez@u) = Lz*np.pi/2")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=0.5) # Random noise
u.low_pass_filter(scales=0.5)
u['g'][0] += u0['g'][0]

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=5, max_writes=1, parallel='gather')
snapshots.add_tasks(solver.state)
slices = solver.evaluator.add_file_handler('slices', sim_dt=0.1, max_writes=100, parallel='gather')
slices.add_task(u(z=0), name='u(z=0)')
slices.add_task(u(phi=0), name='u(phi=0)')
slices.add_task(u(phi=np.pi), name='u(phi=pi)')
slices.add_task(u(r=0.95), name='u(r=0.95)')
slices.add_task(u(r=0.99), name='u(r=0.99)')
slices.add_task(u1(z=0), name='u1(z=0)')
slices.add_task(u1(phi=0), name='u1(phi=0)')
slices.add_task(u1(phi=np.pi), name='u1(phi=pi)')
slices.add_task(u1(r=0.95), name='u1(r=0.95)')
slices.add_task(u1(r=0.99), name='u1(r=0.99)')
slices.add_task(tau_wall, name='tau_wall')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@u)/2, name='KE')
flow.add_property((u1@u1)/2, name='KE1')
flow.add_property(d3.ave(tau_wall), name='tw')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, max(KE)=%f, max(KE1)=%f, tw=%e' %(solver.iteration, solver.sim_time, timestep, flow.max('KE'), flow.max('KE1'), flow.max('tw')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

