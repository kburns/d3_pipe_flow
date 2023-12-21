

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lz = 4
R = 1
Nz = 8
Nphi = 8
Nr = 32
dtype = np.float64

# Bases
z_coord = d3.Coordinate('z')
disk_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = d3.DirectProduct(z_coord, disk_coords)
dist = d3.Distributor(cyl_coords, dtype=dtype)
z_basis = d3.RealFourier(z_coord, size=Nz, bounds=(0, Lz))
disk_basis = d3.DiskBasis(disk_coords, shape=(Nphi, Nr), radius=R, dtype=dtype)
z, phi, r = dist.local_grids(z_basis, disk_basis)

# Fields
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
problem = d3.LBVP([u, p, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("grad(p) - lap(u) + lift(tau_u) = 4*ez")
problem.add_equation("u(r=R) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

u_true = dist.VectorField(cyl_coords, name='u', bases=(z_basis, disk_basis))
u_true['g'][0] = 1 - r**2
print(np.allclose(u['g'], u_true['g']))
print(np.allclose(p['g'], 0))
