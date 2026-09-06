#ifndef RMO_ROPT_OBSERVER_H
#define RMO_ROPT_OBSERVER_H

#include <iosfwd>

namespace rmo
{

//! One evaluated iterate of a (multilevel) descent method.
struct CycleInfo
{
    unsigned iter      = 0;
    unsigned lac_iter  = 0;
    double   step_size = 0.0;
    double   elapsed   = 0.0;
    bool     coarse    = false;  // step came from a coarse correction
    unsigned level     = 0;

    double residual = 0.0;
    double energy   = 0.0;

    // Coarse condition: reported only by levels that have a coarser one, and 0 on the steps
    // where the condition was not evaluated.
    bool   coarse_cond     = false;
    double grad_norm       = 0.0;  // ||g||   on level
    double grad_restr_norm = 0.0;  // ||R g|| on level-1
};


//! Sink for the iteration history of a solver.
class IterationObserver
{
public:
    virtual ~IterationObserver() = default;

    //! One evaluated iterate.
    virtual void add(const CycleInfo&) = 0;

    //! A cycle on `level` starts; called again on every revisit in a W-cycle.
    virtual void begin_level(unsigned /*level*/) {}

    //! A cycle on `level` finished and its history should be written.
    virtual void end_level(unsigned /*level*/, std::ostream& /*os*/) {}
};


//! Mixin for solvers reporting their history. Without an observer a solver runs silently.
class ObservableSolver
{
public:
    void set_observer(IterationObserver& observer) { m_observer = &observer; }

protected:
    IterationObserver* m_observer = nullptr;
};

} // namespace rmo

#endif //RMO_ROPT_OBSERVER_H
