//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef RMO_OPTION_TYPES_H
#define RMO_OPTION_TYPES_H
#include <string>
#include <vector>

namespace rmo
{

// TODO: consistent naming of classes
// Every ordering should be compatible to (geometric) multigrid
enum class Ordering
{
    DEFAULT,
    RANDOM,
    CUTHILL_MCKEE,
    KING,
    MIN_DEG
};

enum class BoundaryCondition
{
    NEUMANN,
    DIRICHLET
};

enum class MeshKind
{
    QUADRILATERAL,
    SIMPLEX
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

// TODO: Merge CoarseMetric + SmoothKind -> MetricKind
//       -> use two instances for smoother metric and coarse metric
enum class MetricKind
{
    NONE,   // Galerkin condition not fulfilled
    FROBENIUS,
    MASS,
    ENERGY_ADAPTIVE,
    FISHER_RAO
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
    //bool multilevel;
    unsigned int n_levels;      // number of levels for global refinement
    //unsigned int min_level;     // minimum level for multilevel algorithms
    //unsigned int max_level;     // maximum level for multilevel algorithms
    std::vector<unsigned> v_levels;
};

// Parameters of the FAS cycle itself, which is problem-independent;
// choice of oracles and transfer operators is problem-specific, see e.g. gpe::CoarseModelOptions
struct FAS_Options
{
    double kappa;           // weight for ratio of restricted and coarse gradient
    double eps;             // minimum norm of restricted gradient
    unsigned coarse_every;  // minimum number of fine steps before coarse step is taken
    bool coarse_energy_adaptive;  // solve coarse model with energy-adaptive gradient descent
};

// Fields for gradient computation with inner solver
// TODO: move to descent.h?
struct GradInfo
{
    double residual;
    unsigned num_iter;
    double tolerance;
    double elapsed_time;
};



// ---------- Gross-Pitaevskii problem ----------
namespace gpe
{

// TODO: support more complicated potentials
enum class Potential
{
    SQUARE,
    OPTICAL_LATTICE,
    EXPRESSION      // muparser expression, see GPE_Options::potential_expr
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
    Potential potential;    // used potential V for matrix M_V
    std::string potential_expr;  // expression for Potential::EXPRESSION, in the coordinates x[,y[,z]]
    bool export_solution;   // write incumbent solutions to disk
};

enum class Transport
{
    FROBENIUS,            // Version IV  (orth. proj. Frobenius)
    MASS,                 // Version IV  (orth. proj. Mass)
    DIFFERENTIAL,         // Version VI  (Mixed)
    ADJOINT_RESTRICTION,  // Version V
    ADJOINT_DIFFERENTIAL, // Version III
    ADJOINT_RESTRICTION_FROBENIUS,  // Frobenius-metric counterpart of ADJOINT_RESTRICTION
    ADJOINT_DIFFERENTIAL_FROBENIUS, // Frobenius-metric counterpart of ADJOINT_DIFFERENTIAL
    DIFFERENTIAL_FROBENIUS,         // Frobenius-metric counterpart of DIFFERENTIAL
    // DIFFERENTIAL_MASS,
    // ADJOINT_DIFFERENTIAL_MASS,
};


enum class Interpolate
{
    NONE,
    MASS
};


// Selection of the multilevel components for the GP problem (main_coarse.cc)
// TODO: MetricKind is still defined in rmo (used by OracleBase::get_metric)
struct CoarseModelOptions
{
    MetricKind metric_t;    // type of coarse oracle (shift metric, gradient metric); NONE: single-level
    MetricKind smooth_t;    // type of fine oracle (gradient descent on fine level)
    MetricKind ccond_t;     // metric for evaluating coarse condition
    Transport transport_t;  // type of vector transport
    Interpolate interpol_t; // galerkin condition on linear interpolator
                            // should be consistent with metric_t
};

} // namespace gpe

} // namespace rmo

#endif //RMO_OPTION_TYPES_H