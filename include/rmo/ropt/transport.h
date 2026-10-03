#ifndef RMO_ROPT_TRANSPORT_H
#define RMO_ROPT_TRANSPORT_H

#include <rmo/ropt/interpolate.h>
#include <rmo/option_types.h>

namespace rmo
{

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


class VectorTransportBase
{
public:
    static constexpr auto id = "";
    virtual ~VectorTransportBase() = default;

    /**
     * @brief Prolongs a tangent vector from the coarse grid to the fine grid.
     * @param x_fine The actual fine iterate (defines the target tangent space).
     * @param y_coarse The coarse iterate (defines the source tangent space).
     * @param v_coarse The vector in T_x S_H to be prolonged.
     * @param dst_fine [out] The prolonged vector, guaranteed to be in T_y S_h.
     */
    virtual void vector_prolongation(const Vector<double>& x_fine,
                                     const Vector<double>& y_coarse,
                                     const Vector<double>& v_coarse,
                                     Vector<double>& dst_fine) const = 0;

    /**
     * @brief Restricts a tangent vector from the fine grid to the coarse grid.
     * @param y_coarse The actual coarse iterate (defines the target tangent space).
     * @param x_fine The fine iterate (defines the source tangent space).
     * @param v_fine The vector in T_y S_h to be restricted.
     * @param dst_coarse [out] The restricted vector, guaranteed to be in T_x S_H.
     */
    virtual void vector_restriction(const Vector<double>& y_coarse,
                                    const Vector<double>& x_fine,
                                    const Vector<double>& v_fine,
                                    Vector<double>& dst_coarse) const = 0;
};

} // namespace rmo

#endif //RMO_ROPT_TRANSPORT_H