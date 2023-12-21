

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Re = 1e4
nz = 1
m = 12
Nz = 2 * nz
Nphi = 2 * m
Nr = 64
dtype = np.complex128

# Bases
z_coord = d3.Coordinate('z')
disk_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = d3.DirectProduct(z_coord, disk_coords)
dist = d3.Distributor(cyl_coords, dtype=dtype)
z_basis = d3.Fourier(z_coord, size=Nz, bounds=(0, 2*np.pi), dtype=dtype)
disk_basis = d3.DiskBasis(disk_coords, shape=(Nphi, Nr), radius=1, dtype=dtype)
z, phi, r = dist.local_grids(z_basis, disk_basis)

# Fields
s = dist.Field(name='s')
u0 = dist.VectorField(cyl_coords, name='u0', bases=disk_basis.radial_basis)
u = dist.VectorField(cyl_coords, name='u', bases=(z_basis, disk_basis))
p = dist.Field(name='p', bases=(z_basis, disk_basis))
tau_u = dist.VectorField(cyl_coords, name='tau_u', bases=(z_basis, disk_basis.edge))
tau_p = dist.Field(name='tau_p')

# Substitutions
dt = lambda A: s*A
lift_basis = disk_basis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Background
u0['g'][0] = 1 - r**2

# Problem
problem = d3.EVP([u, p, tau_u, tau_p], eigenvalue=s, namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + u0@grad(u) + u@grad(u0) + grad(p) - (1/Re)*lap(u) + lift(tau_u) = 0")
problem.add_equation("u(r=1) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver()
sp = solver.subproblems_by_group[(nz, m, None)]
solver.solve_dense(sp)
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
evals = evals[np.argsort(-evals.real)]
for i in range(6):
    print(evals[i])

