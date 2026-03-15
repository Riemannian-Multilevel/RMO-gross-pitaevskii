//
// Created by Ferdinand Vanmaele on 12.01.26.
//

#ifndef GPE_OPTION_TYPES_H
#define GPE_OPTION_TYPES_H

namespace gpe
{
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

enum class VectorTransportKind  // ~Method?
{
    PROJECTION,         // composition of orthogonal projection and linear interpolation
    DIFF,               // differential of prolongation and restriction maps
    DIFF_ADJ_COARSE,    // metric adjoint w.r.t. A/M-metric, starting from prolongation map
    PINV_ADJ_COARSE,    // as above, but taking the pseudo-inverse Dp+
    DIFF_ADJ_FINE,      // metric adjoint w.r.t. A/M-metric, starting from restriction map
    PINV_ADJ_FINE,      // as above, but taking the pseudo-inverse Dr+
};

// TODO: separate inner solver + line search options from (gradient) descent options
struct DescentOptions
{
    double tol_lambda;          // tolerance for rayleigh quotients
    double tol_residual;        // tolerance for M-residual
    double step_size;           // fixed step-size used in iteration steps
    unsigned int max_iter;      // maximum GD iterations
    bool line_search;           // determine step-size by line search

    // TODO: move to separate struct
    unsigned int ls_max_iter;    // maximum line search iterations
    double ls_alpha;
    double ls_beta;
    double ls_sigma;
    double ls_min;
};

struct SolverOptions
{
    unsigned int max_inner;     // maximum sparse solver iterations
    double tol_inner;           // relative tolerance for inner solver
    SolverMethod solver;        // method for solving sparse linear equations
    Precondition precond;       // preconditioner for solving sparse linear equations
};

struct LineSearchOptions
{

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

}

#endif //GPE_OPTION_TYPES_H