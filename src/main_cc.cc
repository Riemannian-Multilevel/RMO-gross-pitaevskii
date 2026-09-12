//
// Single-level Riemannian gradient descent baseline for the continuous-cuts
// problem, on a synthetic image (bright disk on a dark background) with a
// foreground/background seed patch known to lie inside/outside it -- a
// stand-in for a loaded image and its user-provided seed regions (cf.
// Fig. 12). Regularization parameters match the paper's fine-level
// continuous-cuts experiment (Sec. 6.3.4). Reproduces a single-level "RGD"
// run (Fig. 13) as a regression baseline ahead of the multilevel driver
// (still open, see doc/plan_continuous_cuts.tex).
//
#include <rmo/cc/cc.h>
#include <rmo/cc/grid.h>
#include <rmo/cc/manifold.h>
#include <rmo/cc/oracle.h>
#include <rmo/cc/synthetic.h>

#include <rmo/ropt/observer_table.h>
#include <rmo/ropt/solver.h>

#include <cmath>
#include <iostream>

using namespace rmo;
using namespace rmo::cc;


int main()
{
    constexpr double alpha   = 0.1;    // fine-level regularization weight (Sec. 6.3.4)
    constexpr double epsilon = 1e-4;   // fine-level TV smoothing (Sec. 6.3.4)

    const SyntheticImage img(41);
    const PixelGrid grid(img.rows, img.cols);
    const ForwardDifference D(grid);

    ContinuousCutsFunctional functional(D, grid.to_dof_order(build_rho(img)), alpha, epsilon);
    FisherRaoOracle oracle(functional);
    const BernoulliManifold manifold;

    Vector<double> phi(grid.n_dofs());
    phi = 0.5;   // uniform initialization

    DescentOptions options{};
    options.tol_residual = 1e-6;
    options.step_size    = 1.0;
    options.max_iter     = 300;
    options.line_search  = true;
    options.ls.max_iter  = 50;
    options.ls.alpha     = 1.0;
    options.ls.beta      = 0.5;
    options.ls.sigma     = 1e-4;
    options.ls.min       = 1e-12;

    GradientDescent solver(oracle, manifold, options);

    ConvergenceTableObserver observer;
    solver.set_observer(observer);
    solver.cycle(phi, std::cout);

    return 0;
}
