//
// Created by Ferdinand Vanmaele on 05.04.26.
//

#ifndef GPE_FE_INTERPOLATE_H
#define GPE_FE_INTERPOLATE_H

#include <gpe/lac.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

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
                                SparsityPattern& sparsity_pattern,
                                SparseMatrix<double>& interpolation_matrix)
    {
        const unsigned int n_src = dof_src.n_dofs();
        const unsigned int n_dst = dof_dst.n_dofs();

        DynamicSparsityPattern dsp(n_dst, n_src);

        struct MatrixEntry {
            global_dof_index row;
            global_dof_index col;
            double value;
        };

        // 1. Lock-free storage: A dedicated vector for every column.
        // Threads write to column_entries[j] concurrently without needing mutexes.
        std::vector<std::vector<MatrixEntry>> column_entries(n_src);

        // --- THE PARALLEL PROBING LOOP ---
        tbb::parallel_for(tbb::blocked_range<unsigned int>(0, n_src),
            [&](const tbb::blocked_range<unsigned int>& range) {

                // 2. Thread-local memory: Every thread needs its own vectors
                // so they don't overwrite each other's data during interpolation.
                Vector<double> local_unit_src(n_src);
                Vector<double> local_column_dst(n_dst);

                for (unsigned int j = range.begin(); j != range.end(); ++j) {

                    local_unit_src = 0.0;
                    local_unit_src[j] = 1.0;

                    // 3. Thread-safe read operations
                    dealii::VectorTools::interpolate_to_different_mesh(
                        dof_src, local_unit_src, dof_dst, local_column_dst);

                    // Reserve memory to prevent reallocations
                    column_entries[j].reserve(dof_dst.get_fe().dofs_per_cell);

                    for (unsigned int i = 0; i < n_dst; ++i) {
                        if (std::abs(local_column_dst[i]) > 1e-14) {
                            column_entries[j].push_back({i, j, local_column_dst[i]});
                        }
                    }
                }
            });

        std::cerr << "Probing complete. Assembling matrix..." << std::endl;

        // --- SERIAL ASSEMBLY ---
        // 4. Writing to the sparsity pattern and sparse matrix must be done serially.
        // Because the geometry probing is the bottleneck, this serial phase is virtually instant.
        for (unsigned int j = 0; j < n_src; ++j) {
            for (const auto& entry : column_entries[j]) {
                dsp.add(entry.row, entry.col);
            }
        }

        sparsity_pattern.copy_from(dsp);
        interpolation_matrix.reinit(sparsity_pattern);

        for (unsigned int j = 0; j < n_src; ++j) {
            for (const auto& entry : column_entries[j]) {
                interpolation_matrix.set(entry.row, entry.col, entry.value);
            }
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


/**
 * @brief Linear transfer built on deal.II's geometric global-coarsening
 * multigrid infrastructure (MGTwoLevelTransfer).
 *
 * Unlike LinearTransferMatrix, which probes the interpolation matrices
 * column-by-column through VectorTools::interpolate_to_different_mesh (an
 * O(n_dofs) sequence of mesh interpolations, expensive enough to require
 * disk caching), this class lets MGTwoLevelTransfer assemble the two-level
 * transfer from the local element embedding matrices in a single sweep --
 * no probing, no caching. It works directly with the per-level DoFHandlers
 * (each on its own Triangulation, as set up by ModelBuilder), which is
 * exactly the "global coarsening" setting the deal.II infrastructure is
 * designed for; the level variants in FeSpaceMG/assemble.h are the same
 * foundation for the single-DoFHandler case.
 *
 * Operator mapping (cf. the class documentation of MGTwoLevelTransferBase,
 * which distinguishes restriction of "right hand side vectors" from
 * interpolation of "solution vectors"):
 *  - to_fine_mesh()   = prolongate_and_add()  (FE embedding I_H^h)
 *  - to_coarse_mesh() = interpolate()         (solution/primal restriction,
 *                                              the FAS injection r)
 *  - Tfine()          = restrict_and_add()    (exact transpose (I_H^h)^T)
 */
template <int dim>
class LinearTransferMG : public LinearTransferBase
{
    using DVector = dealii::LinearAlgebra::distributed::Vector<double>;

public:
    LinearTransferMG(const dealii::DoFHandler<dim>& dof_coarse,
                     const dealii::DoFHandler<dim>& dof_fine,
                     const dealii::AffineConstraints<double>& constraints_coarse,
                     const dealii::AffineConstraints<double>& constraints_fine)
        : n_c(dof_coarse.n_dofs())
        , n_f(dof_fine.n_dofs())
        , constraints_c(constraints_coarse)
        , constraints_f(constraints_fine)
    {
        transfer.reinit(dof_fine, dof_coarse, constraints_fine, constraints_coarse);
    }

    /**
     * @brief Prolongates a vector from the coarse mesh to the fine mesh.
     * Evaluates $v_{fine} = I_H^h \cdot v_{coarse}$.
     */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        DVector src(n_c), dst(n_f);
        std::copy(src_coarse.begin(), src_coarse.end(), src.begin());

        transfer.prolongate_and_add(dst, src);

        dst_fine.reinit(n_f);
        std::copy(dst.begin(), dst.end(), dst_fine.begin());
        constraints_f.distribute(dst_fine);
    }

    /**
     * @brief Restricts a vector from the fine mesh to the coarse mesh via
     * solution interpolation (pointwise injection at coarse nodes for nested
     * Lagrange elements), matching the point-restriction map r.
     */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        DVector src(n_f), dst(n_c);
        std::copy(src_fine.begin(), src_fine.end(), src.begin());

        transfer.interpolate(dst, src);

        dst_coarse.reinit(n_c);
        std::copy(dst.begin(), dst.end(), dst_coarse.begin());
        constraints_c.distribute(dst_coarse);
    }

    /**
     * @brief Transpose of the prolongation, $(I_H^h)^T$ (multigrid residual
     * restriction). Exact adjoint of to_fine_mesh() w.r.t. the Euclidean
     * pairing by construction.
     */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        DVector src(n_f), dst(n_c);
        std::copy(src_fine.begin(), src_fine.end(), src.begin());

        transfer.restrict_and_add(dst, src);

        dst_coarse.reinit(n_c);
        std::copy(dst.begin(), dst.end(), dst_coarse.begin());
    }

    unsigned int n_coarse() const override { return n_c; }
    unsigned int n_fine() const override { return n_f; }

private:
    unsigned int n_c;
    unsigned int n_f;
    const dealii::AffineConstraints<double>& constraints_c;
    const dealii::AffineConstraints<double>& constraints_f;

    dealii::MGTwoLevelTransfer<dim, DVector> transfer;
};


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

    /**
     * @brief Computes mass-weighted restriction: I_h^H = M_H^{-1} * (I_H^h)^T * M_h
     */
    void to_coarse_mesh(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        // 1. Multiply by fine mass matrix: M_h * v_h
        Vector<double> Mh_v(_transfer.n_fine());
        _M_fine.vmult(Mh_v, src_fine);

        // 2. Apply TRANSPOSE of prolongation: (I_H^h)^T * (M_h * v_h)
        Vector<double> Ih_Mh_v(_transfer.n_coarse());
        _transfer.Tfine(Mh_v, Ih_Mh_v); // FIXED: Was _transfer.to_coarse_mesh

        // 3. Apply inverse coarse mass matrix
        _M_inv_coarse.vmult(dst_coarse, Ih_Mh_v);
    }

    /**
     * @brief Prolongation remains unchanged: I_H^h
     */
    void to_fine_mesh(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        _transfer.to_fine_mesh(src_coarse, dst_fine);
    }

    /**
     * @brief Transpose of prolongation remains unchanged: (I_H^h)^T
     */
    void Tfine(const Vector<double>& src_fine, Vector<double>& dst_coarse) const override
    {
        _transfer.Tfine(src_fine, dst_coarse);
    }

    /**
     * @brief Transpose of mass-weighted restriction: (I_h^H)^T = M_h * I_H^h * M_H^{-1}
     */
    void Tcoarse(const Vector<double>& src_coarse, Vector<double>& dst_fine) const override
    {
        // 1. Multiply by inverse coarse mass matrix: M_H^{-1} * v_H
        Vector<double> MH_inv_v(_transfer.n_coarse());
        _M_inv_coarse.vmult(MH_inv_v, src_coarse);

        // 2. Apply FORWARD prolongation: I_H^h * (M_H^{-1} * v_H)
        Vector<double> TIh_MH_inv_v(_transfer.n_fine());
        _transfer.to_fine_mesh(MH_inv_v, TIh_MH_inv_v); // FIXED: Was _transfer.Tcoarse

        // 3. Multiply by fine mass matrix
        _M_fine.vmult(dst_fine, TIh_MH_inv_v);
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
