//
// Created by Ferdinand Vanmaele on 05.04.26.
//

#ifndef GPE_FE_INTERPOLATE_H
#define GPE_FE_INTERPOLATE_H

#include <gpe/lac.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>


namespace gpe
{

class LinearTransferBase
{
public:
    virtual ~LinearTransferBase() = default;

    // TODO: match vmult() interface (output&, const input&)
    virtual void to_coarse_mesh(const Vector<double>&, Vector<double>&) const = 0;

    virtual void to_fine_mesh(const Vector<double>&, Vector<double>&) const = 0;

    // Transpose of to_coarse_mesh() (for matrix implementations)
    virtual void Tcoarse(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    // Transpose of to_fine_mesh()
    virtual void Tfine(const Vector<double>&, Vector<double>&) const
    {
        throw dealii::ExcNotImplemented(__PRETTY_FUNCTION__);
    }

    virtual unsigned n_coarse() const = 0;
    virtual unsigned n_fine() const = 0;
};


// Linear interpolation using deal.ii mesh interpolation
template <int dim>
class LinearTransfer : public LinearTransferBase
{
public:
    LinearTransfer(const dealii::DoFHandler<dim>& dof_c,
                   const dealii::DoFHandler<dim>& dof_f,
                   const dealii::AffineConstraints<double>& aff_c,
                   const dealii::AffineConstraints<double>& aff_f)
        : dof_coarse(dof_c), dof_fine(dof_f), aff_coarse(aff_c), aff_fine(aff_f)
    {}

    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        dst_coarse.reinit(dof_coarse.n_dofs());
        dst_coarse = 0.0;

        dealii::VectorTools::interpolate_to_coarser_mesh(dof_fine,
            src_fine, dof_coarse, aff_coarse, dst_coarse);
    }

    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        dst_fine.reinit(dof_fine.n_dofs());
        dst_fine = 0.0;

        dealii::VectorTools::interpolate_to_finer_mesh(dof_coarse,
            src_coarse, dof_fine, aff_fine, dst_fine);
    }

    unsigned n_coarse() const override { return dof_coarse.n_dofs(); }
    unsigned n_fine() const override { return dof_fine.n_dofs(); }

private:
    const dealii::DoFHandler<dim>& dof_coarse;
    const dealii::DoFHandler<dim>& dof_fine;
    const dealii::AffineConstraints<double>& aff_coarse;
    const dealii::AffineConstraints<double>& aff_fine;
};


template <int dim>
class LinearTransferMatrix : public LinearTransferBase
{
public:
    /**
     * @brief Constructs the transfer operators between two given DoF handlers.
     * @param dof_coarse The DoFHandler for the coarse grid.
     * @param dof_fine The DoFHandler for the fine grid.
     * @param constraints_coarse
     * @param constraints_fine
     */
    LinearTransferMatrix(const dealii::DoFHandler<dim>& dof_coarse,
                         const dealii::DoFHandler<dim>& dof_fine,
                         const dealii::AffineConstraints<double>& constraints_coarse,
                         const dealii::AffineConstraints<double>& constraints_fine)
        : n_c(dof_coarse.n_dofs())
        , n_f(dof_fine.n_dofs())
        , constraints_c(constraints_coarse) // Store references
        , constraints_f(constraints_fine)
    {
        // Prolongation P: Maps Coarse -> Fine
        //build_interpolation_matrix(dof_coarse, dof_fine, sparsity_pattern_P, P);

        // Restriction R: Maps Fine -> Coarse
        //build_interpolation_matrix(dof_fine, dof_coarse, sparsity_pattern_R, R);

        // Create unique filenames based on the grid sizes to prevent mismatched loads
        const std::string p_filename = "P_matrix_" + std::to_string(n_c) + "_to_" + std::to_string(n_f) + ".bin";
        const std::string r_filename = "R_matrix_" + std::to_string(n_f) + "_to_" + std::to_string(n_c) + ".bin";

        // Prolongation P: Maps Coarse -> Fine
        load_or_build_matrix(p_filename, dof_coarse, dof_fine, sparsity_pattern_P, P);

        // Restriction R: Maps Fine -> Coarse
        load_or_build_matrix(r_filename, dof_fine, dof_coarse, sparsity_pattern_R, R);
    }

    /**
     * @brief Prolongates a vector from the coarse mesh to the fine mesh.
     * Evaluates $v_{fine} = P \cdot v_{coarse}$.
     * * @param src_coarse The input vector on the coarse grid.
     * @param dst_fine [out] The interpolated vector on the fine grid.
     */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        P.vmult(dst_fine, src_coarse);

        constraints_f.distribute(dst_fine);
    }

    /**
     * @brief Restricts a vector from the fine mesh to the coarse mesh via point interpolation.
     * Evaluates $v_{coarse} = R \cdot v_{fine}$.
     * * @param src_fine The input vector on the fine grid.
     * @param dst_coarse [out] The restricted vector on the coarse grid.
     */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        R.vmult(dst_coarse, src_fine);

        constraints_c.distribute(dst_coarse);
    }

    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        P.Tvmult(dst_coarse, src_fine);

        constraints_c.distribute(dst_coarse);  // needed?
    }

    void Tcoarse(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        R.Tvmult(dst_fine, src_coarse);

        constraints_f.distribute(dst_fine);
    }

    /** @brief Returns the number of degrees of freedom on the coarse grid. */
    unsigned int n_coarse() const override { return n_c; }

    /** @brief Returns the number of degrees of freedom on the fine grid. */
    unsigned int n_fine() const override { return n_f; }

    const SparseMatrix<double>& get_P() { return P; }
    const SparseMatrix<double>& get_R() { return R; }

private:
    unsigned int n_c;
    unsigned int n_f;
    dealii::AffineConstraints<double> constraints_c, constraints_f;

    SparsityPattern sparsity_pattern_P;
    SparseMatrix<double> P;

    SparsityPattern sparsity_pattern_R;
    SparseMatrix<double> R;

    void build_interpolation_matrix(const dealii::DoFHandler<dim>& dof_src,
                                    const dealii::DoFHandler<dim>& dof_dst,
                                    dealii::SparsityPattern& sparsity_pattern,
                                    dealii::SparseMatrix<double>& interpolation_matrix)
    {
        const unsigned int n_src = dof_src.n_dofs();
        const unsigned int n_dst = dof_dst.n_dofs();

        // Matrix maps from src -> dst, so dimensions are (n_dst) x (n_src)
        dealii::DynamicSparsityPattern dsp(n_dst, n_src);

        struct MatrixEntry {
            global_dof_index row;
            global_dof_index col;
            double value;
        };
        std::vector<MatrixEntry> cached_entries;

        // Estimate ~9-27 non-zeros per column depending on dimension
        cached_entries.reserve(n_src * dof_dst.get_fe().dofs_per_cell);

        Vector<double> unit_src(n_src);
        Vector<double> column_dst(n_dst);

        // --- THE PROBING LOOP ---
        // We push a 1.0 through each source DoF. The resulting interpolated
        // destination vector is exactly one column of the transfer matrix.
        for (unsigned int j = 0; j < n_src; ++j) {
            if (j % 1000 == 0) std::cerr << j << "..";

            // 1. Create the unit vector
            unit_src = 0.0;
            unit_src[j] = 1.0;

            // 2. Let deal.II handle all the nasty constraint/boundary math!
            dealii::VectorTools::interpolate_to_different_mesh(
            dof_src, unit_src, dof_dst, column_dst);

            // 3. Record the non-zero entries for this column
            for (unsigned int i = 0; i < n_dst; ++i) {
                if (std::abs(column_dst[i]) > 1e-14) {
                    dsp.add(i, j);
                    cached_entries.push_back({i, j, column_dst[i]});
                }
            }
        }
        std::cerr << std::endl;
        // 4. Initialize and fill the sparse matrix
        sparsity_pattern.copy_from(dsp);
        interpolation_matrix.reinit(sparsity_pattern);

        for (const auto& entry : cached_entries) {
            interpolation_matrix.set(entry.row, entry.col, entry.value);
        }
    }

    void load_or_build_matrix(const std::string& filename,
                              const dealii::DoFHandler<dim>& dof_src,
                              const dealii::DoFHandler<dim>& dof_dst,
                              dealii::SparsityPattern& sparsity_pattern,
                              dealii::SparseMatrix<double>& matrix)
    {
        // Try to open the file in binary mode
        std::ifstream in_file(filename, std::ios::binary);

        if (in_file.is_open()) {
            std::cerr << "  -> Loading transfer matrix from disk: " << filename << std::endl;

            // 1. Read and finalize the SparsityPattern first
            sparsity_pattern.block_read(in_file);

            // 2. Initialize the empty matrix with the loaded pattern
            matrix.reinit(sparsity_pattern);

            // 3. Read the actual matrix values (the float data)
            matrix.block_read(in_file);

        } else {
            std::cerr << "  -> Building transfer matrix from scratch..." << std::endl;

            // 1. Call expensive probing/geometric setup function
            build_interpolation_matrix(dof_src, dof_dst, sparsity_pattern, matrix);

            // 2. Open file for writing in binary mode
            std::ofstream out_file(filename, std::ios::binary);

            if (out_file.is_open()) {
                // Write the pattern first, then the matrix
                sparsity_pattern.block_write(out_file);
                matrix.block_write(out_file);
                std::cerr << "  -> Saved transfer matrix to: " << filename << std::endl;
            } else {
                std::cerr << "  -> Warning: Could not open file to save matrix: " << filename << std::endl;
            }
        }
    }
};


// TODO: inherits from LinearTransferBase, yet takes a LinearTransferBase as an argument....?
template <int dim, typename TransferType, typename MatrixType, typename InverseMatrixType>
class MassTransfer : public LinearTransferBase
{
public:
    MassTransfer(const dealii::DoFHandler<dim>& dof_coarse,
                 const dealii::DoFHandler<dim>& dof_fine,
                 const dealii::AffineConstraints<double>& constraints_coarse,
                 const dealii::AffineConstraints<double>& constraints_fine,
                 const MatrixType& M_fine,
                 const InverseMatrixType& M_inv_coarse)
        : _transfer(TransferType(dof_coarse, dof_fine, constraints_coarse, constraints_fine))
        , _M_fine(M_fine)
        , _M_inv_coarse(M_inv_coarse)
    {}

    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        Vector<double> Mh_v(_transfer.n_fine());
        _M_fine.vmult(Mh_v, src_fine);

        Vector<double> Ih_Mh_v(_transfer.n_coarse());
        _transfer.to_coarse_mesh(Mh_v, Ih_Mh_v);

        _M_inv_coarse.vmult(dst_coarse, Ih_Mh_v);
    }

    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        _transfer.to_fine_mesh(src_coarse, dst_fine);
    }

    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        _transfer.Tfine(dst_coarse, src_fine);
    }

    void Tcoarse(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        Vector<double> MH_inv_v(_transfer.n_coarse());
        _M_inv_coarse.Tvmult(MH_inv_v, src_coarse);  // symmetric

        Vector<double> TIh_MH_inv_v(_transfer.n_fine());
        _transfer.Tcoarse(MH_inv_v, TIh_MH_inv_v);

        _M_fine.vmult(dst_fine, TIh_MH_inv_v);       // symmetric
    }

    unsigned n_coarse() const override { return _transfer.n_coarse(); };
    unsigned n_fine() const override { return _transfer.n_fine(); };

private:
    TransferType _transfer;
    const MatrixType& _M_fine;
    const InverseMatrixType& _M_inv_coarse;
};

} // namespace gpe

#endif //GPE_FE_INTERPOLATE_H
