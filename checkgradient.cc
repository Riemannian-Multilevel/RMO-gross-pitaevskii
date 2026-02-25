//
// Created by Ferdinand Vanmaele on 24.02.26.
//
#include "gpe.h"
#include "random.h"
#include "manifold.h"
#define NUM_TRIALS 100

using namespace gpe;

namespace mass
{
    template <typename MatrixType>
    void random_tangent_vector(const Vector<double>& x, const MatrixType& M, Vector<double>& output)
    {
        static double mean   = 0.0;
        static double stddev = 1.0;

        // 1. generate random vector in ambient space
        Vector<double> tmp(output.size());
        normrnd(mean, stddev, tmp);

        // 2. project orthogonally onto tangent space at x (in sphere S, using M-metric)
        project_onto_tangent_space(x, M, tmp, output);
    }
}

template <typename Oracle>
double finite_difference(const Oracle& O, const Vector<double>& x,
    const Vector<double>& v, double h)
{
    AssertThrow(h > 0, dealii::ExcMessage("h must be positive"));

    // y <- x + hv, v direction
    Vector<double> tmp(x);
    tmp.add(h, v);

    // [ O(x+hv) - O(x) ] / h
    return (O.value(tmp) - O.value(x)) / h;
}

int main()
{
    GPE_Options options{};
    options.dimension = 2;
    options.degree    = 1;  // piecewise linear (1) or quadratic (2) elements
    options.radius    = 10;
    options.beta      = 100;
    options.bc        = BoundaryCondition::DIRICHLET;
    options.mesh_kind = MeshKind::QUADRILATERAL;
    options.order     = Ordering::CUTHILL_MCKEE;

    // GPE minimization
    constexpr unsigned int n_levels = 8;
    GrossPitaevskiiPackage<2> GS(options, n_levels);

    // The Riemannian gradient of E at x is defined as,
    // the unique element in the tangent space at x, T_x S,
    // such that for all v in T_x S,
    // g_x(\grad(x), v) = DE(x)[v]

    for (unsigned int trial = 0; trial < NUM_TRIALS; trial++) {

    }
}