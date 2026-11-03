"""
Pipe flow stability eigenvalue problem in Dedalus.
With radius R and centerline velocity U, the Reynolds number is Re = R*U/nu.
This script is nondimensionalized such that R=1 and U=1.
The base parabolic flow profile is then u0(r) = 1 - r**2.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
kz = 1
m = 12
Nr = 64
Re = 3e3
n_evals = 6
dtype = np.complex128

# Derived/fixed parameters
Nphi = 2 * (abs(m) + 1)  # includes azimuthal wavenumbers up to |m|
Nz = 3  # wavenumbers [0, |kz|, -|kz|]
Lz = 2 * np.pi / (abs(kz) + (kz==0))

# Bases
z_coord = d3.Coordinate('z')
disk_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = d3.DirectProduct(z_coord, disk_coords)
dist = d3.Distributor(cyl_coords, dtype=dtype)
z_basis = d3.Fourier(z_coord, size=Nz, bounds=(0, Lz), dtype=dtype)
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
nz = int(np.sign(kz))  # z group: -1, 0, or 1
solver = problem.build_solver()
sp = solver.subproblems_by_group[(nz, m, None)]
solver.solve_dense(sp)
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
evals = evals[np.argsort(-evals.real)]
for i in range(n_evals):
    print(evals[i])

