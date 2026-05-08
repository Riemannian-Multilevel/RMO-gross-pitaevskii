//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_OPTION_TYPES_H
#define GPE_OPTION_TYPES_H

namespace gpe
{

// TODO: consistent naming of classes
// Every ordering should be compatible to (geometric) multigrid
enum class Ordering
{
    DEFAULT,
    RANDOM,
    CUTHILL_MCKEE
};

enum class BoundaryCondition
{
    NEUMANN,
    DIRICHLET
};

enum class SolverMethod
{
    GMRES,
    MINRES,
    CG
};

enum class Precondition
{
    NONE,
    DIAGONAL,
    SPARSE_ILU,
    AMG
};

enum class MeshKind
{
    QUADRILATERAL,
    SIMPLEX
};

// TODO: support more complicated potentials
enum class Potential
{
    SQUARE
};

// TODO: Merge CoarseMetric + SmoothKind -> MetricKind
//       -> use two instances for smoother metric and coarse metric
enum class MetricKind
{
    NONE,   // Galerkin condition not fulfilled
    FROBENIUS,
    MASS,
    ENERGY_ADAPTIVE
};

enum class Transport
{
    FROBENIUS,
    MASS,
    DIFFERENTIAL,
    ADJOINT_RESTRICTION,
    ADJOINT_RESTRICTION_SCALED,
};

enum class Interpolate
{
    NONE,
    MASS
};


// ----- Option structures
struct DescentOptions
{
    double tol_lambda;          // tolerance for rayleigh quotients
    double tol_residual;        // tolerance for M-residual
    double step_size;           // fixed step-size used in iteration steps
    unsigned int max_iter;      // maximum GD iterations
    bool line_search;           // determine step-size by line search

    struct LineSearchOptions
    {
        unsigned int max_iter;    // maximum line search iterations
        double alpha;             // starting step-size for backtracking
        double beta;              // reduction factor for backtracking
        double sigma;             // factor for sufficient decrease
        double min;               // minimum step-size taken
    } ls;
};

struct SolverOptions
{
    unsigned int max_inner;     // maximum sparse solver iterations
    double tol_inner;           // relative tolerance for inner solver
    double tol_inner_res;       // relative tolerance for inner solver, w.r.t residual
    SolverMethod solver;        // method for solving sparse linear equations
    Precondition precond;       // preconditioner for solving sparse linear equations
};

struct MG_Options
{
    bool multilevel;            // build a multilevel hierarchy
    unsigned int n_levels;      // number of levels for global refinement
    unsigned int min_level;     // minimum level for multilevel algorithms
    unsigned int max_level;     // maximum level for multilevel algorithms
};

struct GPE_Options
{
    int dimension;          // dimension of domain
    int degree;             // degree of shape functions
    double radius;          // radius of the cube (square, line) domain
    double beta;            // factor for the non-linear term in GPE
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
    MeshKind mesh_kind;     // subdivide the grid into simplices or quadrilaterals
};

struct FAS_Options
{
    double kappa;           // weight for ratio of restricted and coarse gradient
    double eps;             // minimum norm of restricted gradient
    unsigned coarse_every;  // minimum number of fine steps before coarse step is taken
    bool coarse_energy_adaptive;  // solve coarse model with energy-adaptive gradient descent

    MetricKind metric_t;    // type of coarse oracle (shift metric, gradient metric)
    MetricKind smooth_t;    // type of fine oracle (gradient descent on fine level)
    Transport transport_t;  // type of vector transport
    Interpolate interpol_t; // galerkin condition on linear interpolator
                            // should be consistent with metric_t
};

} // namespace gpe

#endif //GPE_OPTION_TYPES_H