#ifndef RMO_ROPT_TRANSPORT_H
#define RMO_ROPT_TRANSPORT_H

#include <rmo/ropt/interpolate.h>
#include <rmo/ropt/metric.h>
#include <rmo/option_types.h>

/**
 * @file
 * @brief Interfaces of the transfers between a fine manifold \f$ \mathcal{S}_h \f$ and a coarse manifold
 * \f$ \mathcal{S}_H \f$ (implementations and notation: gpe/transport.h).
 */
namespace rmo
{

/** @brief Point maps \f$ r: \mathcal{S}_h \to \mathcal{S}_H \f$ and \f$ p: \mathcal{S}_H \to \mathcal{S}_h \f$, built on a linear transfer. */
class ManifoldTransferBase
{
public:
    ManifoldTransferBase(const LinearTransferBase& transfer_)
        : transfer(transfer_) {}
    virtual ~ManifoldTransferBase() = default;

    // Mandatory operations
    virtual void restriction(const Vector<double>&, Vector<double>&) const = 0;

    virtual void prolongation(const Vector<double>&, Vector<double>&) const = 0;

    // Optional operations
    virtual void diff_restriction(const Vector<double>&, const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented("Differential not implemented for manifold transfer");
    }

    virtual void diff_prolongation(const Vector<double>&, const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented("Differential not implemented for manifold transfer");
    }

    //! \f$ \mathrm{dst} = \mathrm{D}p(y)^* u = M_H^{-1} \mathrm{D}p(y)^\top M_h u \f$, the \f$ M \f$-adjoint of
    //! diff_prolongation() at @p y applied to @p u (point transfers that hold \f$ M_H^{-1} \f$)
    virtual void diff_prolongation_adjoint(const Vector<double>&, const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented("Adjoint differential not implemented for manifold transfer");
    }

    // Dimension
    [[nodiscard]] unsigned n_fine() const { return transfer.n_fine(); }
    [[nodiscard]] unsigned n_coarse() const { return transfer.n_coarse(); }

protected:
    // For now, every manifold transfer assumes an underlying (linear) interpolation for the embedding space
    const LinearTransferBase& transfer;
};


/** @brief Transport of tangent vectors between the coarse and the fine manifold. */
class VectorTransportBase
{
public:
    static constexpr auto id = "";
    virtual ~VectorTransportBase() = default;

    /**
     * @brief Prolongs @p v_coarse \f$ \in T_y \mathcal{S}_H \f$ at the coarse iterate \f$ y \f$ (@p y_coarse) to
     * @p dst_fine \f$ \in T_x \mathcal{S}_h \f$ at the fine iterate \f$ x \f$ (@p x_fine).
     */
    virtual void vector_prolongation(const Vector<double>& x_fine,
                                     const Vector<double>& y_coarse,
                                     const Vector<double>& v_coarse,
                                     Vector<double>& dst_fine) const = 0;

    /**
     * @brief Restricts @p v_fine \f$ \in T_x \mathcal{S}_h \f$ at the fine iterate \f$ x \f$ (@p x_fine) to
     * @p dst_coarse \f$ \in T_y \mathcal{S}_H \f$ at the coarse iterate \f$ y \f$ (@p y_coarse).
     */
    virtual void vector_restriction(const Vector<double>& y_coarse,
                                    const Vector<double>& x_fine,
                                    const Vector<double>& v_fine,
                                    Vector<double>& dst_coarse) const = 0;
};


/**
 * @brief An adjoint transport whose restriction first projects its argument onto \f$ T_x \f$, orthogonally in the
 * metric \f$ G \f$ of the adjoint: \f$ \mathcal{R} g = \mathcal{R}_{inner} \Pi^G_x g \f$.
 *
 * The adjoint transports compute \f$ \mathcal{R} = \Pi_y \mathrm{D}p(y)^* \f$, the \f$ G \f$-adjoint of
 * \f$ \mathcal{P} = \Pi^G_x \mathrm{D}p(y) \f$ only on \f$ T_x \f$; on the whole space the adjoint is
 * \f$ \Pi_y \mathrm{D}p(y)^* \Pi^G_x \f$, since \f$ \Pi^G_x \f$ is \f$ G \f$-self-adjoint. Computed gradients have a
 * normal component of relative size \f$ \sim \epsilon_{mach} \lambda / \|g\| \f$ near convergence, so without the
 * projection the Galerkin condition holds only up to it (md/positive_multilevel_benchmark.md, step 3c). The projection
 * must be orthogonal in the metric of the adjoint, whatever the metric of the gradient: the metric of the Galerkin
 * condition, i.e. of the coarse model (the tilt oracles of CoarseOracleBase), which the caller chooses together with
 * the transport. Cost: one projection per restriction.
 */
class TangentProjectedTransport : public VectorTransportBase
{
public:
    static constexpr const char* id = "TP";

    TangentProjectedTransport(const VectorTransportBase& inner, const TangentProjectionBase& fine_projection)
        : inner(inner), fine_projection(fine_projection)
    {}

    void vector_prolongation(const Vector<double>& x_fine, const Vector<double>& y_coarse,
                             const Vector<double>& v_coarse, Vector<double>& dst_fine) const override
    {
        inner.vector_prolongation(x_fine, y_coarse, v_coarse, dst_fine);
    }

    void vector_restriction(const Vector<double>& y_coarse, const Vector<double>& x_fine,
                            const Vector<double>& v_fine, Vector<double>& dst_coarse) const override
    {
        Vector<double> v_tangent(v_fine.size());
        fine_projection.project(x_fine, v_fine, v_tangent);
        inner.vector_restriction(y_coarse, x_fine, v_tangent, dst_coarse);
    }

private:
    const VectorTransportBase& inner;
    const TangentProjectionBase& fine_projection;
};

} // namespace rmo

#endif //RMO_ROPT_TRANSPORT_H