import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import dx, grad, inner, dot, div  # ds
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import pyvista
import matplotlib.pyplot as plt

plt.close("all")

a = 100
L = 1
N = a * 1
k = 0.5
t_amb = 0


msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD, nx=a)

V = fem.FunctionSpace(msh, ("Lagrange", 1))
W = fem.FunctionSpace(msh, ("Lagrange", 1))

boundaries = [
    (1, lambda x: np.isclose(x[0], 0)),
    (2, lambda x: np.isclose(x[0], 1)),
]

for marker, locator in boundaries:
    mesh.locate_entities_boundary(msh, 1, marker)
# facet_indices, facet_markers = [], []
# fdim = 1
# for marker, locator in boundaries:
#     facets = mesh.locate_entities(mesh, fdim, locator)
#     facet_indices.append(facets)
#     facet_markers.append(np.full_like(facets, marker))
# facet_indices = np.hstack(facet_indices).astype(np.int32)
# facet_markers = np.hstack(facet_markers).astype(np.int32)
# sorted_facets = np.argsort(facet_indices)
# fi, fm = facet_indices[sorted_facets], facet_markers[sorted_facets]
# facet_tag = mesh.meshtags(mesh, fdim, fi, fm)
# ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)
# facets = mesh.locate_entities_boundary(
#     msh,
#     dim=(msh.topology.dim - 1),
#     marker=lambda x: np.isclose(x[0], 0.0),
# )

# dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
# bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
w = fem.Function(W)
# u_ = fem.Function(V)
u_1 = fem.Function(V)
u_a = fem.Function(V)

w.vector.array[:] = 1.0
u_1.vector.array[:] = 1.0
u_a.vector.array[:] = 0.0


dt = 1 / N
a = 2 * v * u * dx + dt * (v * div(w * grad(u)) * dx) + k * u * v * ds(1)
L = 2 * v * u_1 * dx - dt * (v * div(w * grad(u_1)) * dx) + k * u_a * v * ds(1)


cells, types, xyz = plot.create_vtk_mesh(V)
x = xyz[:, 0]
# u_1.vector.array = f(x)

t = 0
for i in range(10):
    problem = fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    uh = problem.solve()
    t += dt
    y = uh.x.array.real
    u_1.vector.array = y

    fig, ax = plt.subplots()
    plt.plot(x, y)
    plt.title("t={:.2f}".format(t))
    plt.show()

# how to change u_1 properly?
# how to implement a robin BC?
