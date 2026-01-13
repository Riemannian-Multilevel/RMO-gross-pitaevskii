This directory contains old ideas that didn't quite work out as intended.

## `mesh.h`

The solution of the GPE is log-concave and exhibits no singularities. Thus, an
adaptively refined mesh (with resulting increased condition of the mass and
stiffness matrices) did not increase convergence - at least, when created by hand.

Later approaches with Kelly-like (a-posteriori) estimators may still be investigated.


## `mgtools.h`

We can distinguish two approaches for a hierarchical structure of discretizations:

* Geometric multigrid

This is a very specific algorithm which uses pre-/post-smoothing and transfer of
the residuals. deal.ii expects the required conditions to be fulfilled in a very
specific way, and many functions from the single level (active cell) case are
duplicated - e.g. in the MGTools namespace.

Besides added case distinctions, using the same degrees of freedom object for
different hierarchy levels makes visualization of solutions more involved.

* Geometric multilevel

This is the framework in multilevel optimization. Here, we are only looking for
a /transfer/ of solutions between (independent) grids.

While deal.ii does not seem to expose a (sparse) matrix interpolation or
restriction operator, without setting up the geometric multigrid machinery
above, it does have generic interpolation/restriction functions for degrees of
freedom between different meshes, that result from refining/coarsening of a
single mesh.


## `workstream.h`

When using piecewise-linear elements on a globally refined rectangular grid,
assembly of the mass- and stiffness matrices is memory-bound. Therefore we
achieve little by parallelizing the (floating-point) computations on the CPU.

Instead, a matrix-free approach should be used. This also reduces required
memory, since no matrix objects need to be stored.
