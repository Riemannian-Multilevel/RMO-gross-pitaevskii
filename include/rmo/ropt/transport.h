#ifndef RMO_ROPT_TRANSPORT_H
#define RMO_ROPT_TRANSPORT_H

#include <rmo/ropt/interpolate.h>
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

} // namespace rmo

#endif //RMO_ROPT_TRANSPORT_H