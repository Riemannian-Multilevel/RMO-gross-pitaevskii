#ifndef RMO_ROPT_OBSERVER_H
#define RMO_ROPT_OBSERVER_H

#include <iosfwd>

/**
 * @file
 * @brief Observers of the iteration history of the solvers (IterationObserver, ObservableSolver).
 */
namespace rmo
{

//! Sink for the iteration history of a solver.
template <typename InfoType>
class IterationObserver
{
public:
    virtual ~IterationObserver() = default;

    //! One evaluated iterate.
    virtual void add(const InfoType&) = 0;

    //! A cycle on `level` starts; called again on every revisit in a W-cycle.
    virtual void begin_level(unsigned /*level*/) {}

    //! A cycle on `level` finished and its history should be written.
    virtual void end_level(unsigned /*level*/, std::ostream& /*os*/) {}
};


//! Mixin for solvers reporting their history. Without an observer a solver runs silently.
template <typename InfoType>
class ObservableSolver
{
public:
    void set_observer(IterationObserver<InfoType>& observer) { m_observer = &observer; }

protected:
    IterationObserver<InfoType>* m_observer = nullptr;
};

} // namespace rmo

#endif //RMO_ROPT_OBSERVER_H
