"""
Pipe flow base state boundary value problem in Dedalus.
With radius R and centerline velocity U, the Reynolds number is Re = R*U/nu.
This script is nondimensionalized such that R=1 and U=1.
A uniform axial forcing f_z = 4/Re (equivalent to a pressure gradient) then
drives the parabolic flow profile u = (1 - r**2)*ez.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Nphi = 2
Nr = 8
Re = 3e3
dtype = np.float64

# Bases
z_coord = d3.Coordinate('z')
disk_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = d3.DirectProduct(z_coord, disk_coords)
dist = d3.Distributor(cyl_coords, dtype=dtype)
disk_basis = d3.DiskBasis(disk_coords, shape=(Nphi, Nr), radius=1, dtype=dtype)
phi, r = dist.local_grids(disk_basis)

# Fields
u = dist.VectorField(cyl_coords, name='u', bases=disk_basis)
p = dist.Field(name='p', bases=disk_basis)
tau_u = dist.VectorField(cyl_coords, name='tau_u', bases=disk_basis.edge)
tau_p = dist.Field(name='tau_p')

# Substitutions
ez = dist.VectorField(cyl_coords, name='ez', bases=disk_basis)
ez['g'][0] = 1
lift_basis = disk_basis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Problem
problem = d3.LBVP([u, p, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("grad(p) - (1/Re)*lap(u) + lift(tau_u) = (4/Re)*ez")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

# Check against the analytic base flow
u0 = dist.VectorField(cyl_coords, name='u0', bases=disk_basis.radial_basis)
u0['g'][0] = 1 - r**2
print(f"Numerical solution matches reference: {np.allclose(u['g'], u0['g'])}")
