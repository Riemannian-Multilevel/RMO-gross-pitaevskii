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
    DIRICHLET,
    ROBIN
};

enum class SolverMethod
{
    GMRES,
    MINRES,
    CG
};

struct MG_Options
{
    bool multilevel;            // build a multilevel hierarchy
    unsigned int n_levels;      // number of levels for global refinement
    unsigned int min_level;     // minimum level for multilevel algorithms
    unsigned int max_level;     // maximum level for multilevel algorithms
};

struct GdOptions
{
    double tol_inner;       // relative tolerance for inner solver
    double tol_lambda;      // tolerance for rayleigh quotients
    double tol_residual;    // tolerance for M-residual
    double step_size;       // fixed step-size used in iteration steps
    unsigned int max_iter;  // maximum GD iterations
    unsigned int max_inner; // maximum sparse solver iterations
    SolverMethod solver;    // method for solving sparse linear equations
};

struct GPE_Options
{
    int dimension;          // dimension of domain
    int degree;             // degree of shape functions
    double radius;          // radius of the cube (square, line) domain
    double beta;            // factor for the non-linear term in GPE
    Ordering order;         // ordering for degrees of freedom
    BoundaryCondition bc;   // problem boundary conditions (dirichlet or neumann)
};

}
#endif //GPE_OPTION_TYPES_H