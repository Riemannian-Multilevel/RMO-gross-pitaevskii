//
// Created by Ferdinand Vanmaele on 08.04.26.
//
// TODO: move to fas.h
#ifndef GPE_MAIN_COARSE_H
#define GPE_MAIN_COARSE_H

#include <gpe/lac.h>
#include <gpe/problem/oracle.h>
#include <gpe/ropt/transport.h>

namespace gpe
{
template <typename MatrixType>
double M_norm(const MatrixType& M, const Vector<double>& x)
{
    Vector<double> Mx(x.size());
    M.vmult(Mx, x);
    return std::sqrt(x*Mx);
}


struct CycleInfo
{
    unsigned iter;
    unsigned lac_iter;
    double   step_size;
    double   elapsed;
    bool     coarse;
};


// Implementation of smoothing steps
template <typename Oracle>
CycleInfo cycle_smooth(Oracle& O_fine, Vector<double>& x, const Vector<double>& eta,
                       dealii::Timer& timer, DescentOptions options_gd)
{
    timer.start();
    double step_size = options_gd.step_size;

    // Update point (fixed step or line search)
    if (options_gd.line_search) {
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: line search" << std::endl;
#endif
        double Ex = O_fine.value(x);
        double dir_deriv = O_fine.directional_derivative(x, eta);

        step_size = armijo_line_search(O_fine, x, eta, Ex, dir_deriv, options_gd);

        if (step_size <= options_gd.ls.min) {
            std::cerr << "  -> Step rejected by line search." << std::endl;
        }
    }
    else {
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: retraction" << std::endl;
#endif
        // TODO: pass on ManifoldBase object
        ellipsoid::retract_by_norm(O_fine.get_M(), eta, x, options_gd.step_size);  // update y

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] " << "fine: assembly" << std::endl;
#endif
        O_fine.update(x);
    }

    timer.stop();
    return {.step_size = step_size, .elapsed = timer.cpu_time()};
}


template <typename Oracle>
void cycle_eval(const Oracle& O, const Vector<double>& y,
                dealii::ConvergenceTable& convergence_table,
                const CycleInfo info)
{
    auto state   = O.residual(y);
    state.energy = O.value(y);

    convergence_table.add_value("iter", info.iter);
    convergence_table.add_value("coarse", info.coarse ? "*" : " ");
    convergence_table.add_value("lac_iter", info.lac_iter);
    convergence_table.add_value("mass", state.mass);
    convergence_table.add_value("lambda", state.lambda);
    convergence_table.add_value("residual", state.residual);
    convergence_table.add_value("energy", state.energy);
    convergence_table.add_value("step",info.step_size);
    convergence_table.add_value("elapsed",info.elapsed);
}


inline void cycle_finalize(dealii::ConvergenceTable& convergence_table, std::ostream& os,
                           dealii::TableHandler::TextOutputFormat format)
{
    convergence_table.set_precision("mass", 4);
    convergence_table.set_precision("lambda", 4);
    convergence_table.set_precision("residual", 4);
    convergence_table.set_precision("energy", 8);
    convergence_table.set_precision("step", 4);
    convergence_table.set_precision("elapsed", 4);

    convergence_table.set_scientific("lambda", true);
    convergence_table.set_scientific("residual", true);
    convergence_table.set_scientific("energy", true);
    convergence_table.set_scientific("step", true);
    convergence_table.set_scientific("elapsed", true);

    convergence_table.evaluate_convergence_rates("residual", reduction_rate);
    convergence_table.evaluate_convergence_rates("residual", reduction_rate_log2);
    convergence_table.write_text(os, format);
}


struct CoarseStep
{
    CoarseStep(const Vector<double>& y, const unsigned n_coarse, const unsigned n_fine)
        : x(y), y(n_coarse), y_grad(n_coarse), x_grad(n_fine), x_grad_restr(n_coarse)
    {
        AssertDimension(y.size(), n_fine);
    }

    const Vector<double>& x;     // fine point (unchanged in coarse cycle)
    Vector<double> y;            // coarse point
    Vector<double> y_grad;       // (M-)coarse gradient
    Vector<double> x_grad;       // (M-)fine gradient
    Vector<double> x_grad_restr; // restricted gradient
};

// TODO: extend members as needed
struct CoarseCond
{
    double norm_fine;      // (M-)norm of fine gradient
    double norm_coarse;    // (M-)norm of restricted gradient
};


// TODO: TiltOracle template chosen to ensure consistency between O_coarse and O_fine (-> w)
//       use runtime type-check or move functionality to CoarseOracle(Base)
template <int dim>
class CoarseModel
{
public:
    // O_coarse: oracle for evaluating \grad E_c(y) in correction term w = \grad E_c(y) - R \grad E_f(x)
    //           assumed to be consistent with metric in oracle for evaluating <w, .>_y
    // O_fine:   oracle for evaluating \grad E_f(x) in correction term w = \grad E_c(y) - R \grad E_f(x)
    //           assumed to be consistent with metric in oracle for evaluating <w, .>_y
    //           independent of oracle used for gradient descent on the fine level
    // qk:       oracle for evaluating coarse model q_k(y) = E_c(y) + <w, .>_y
    //           oracle for evaluating gradient of coarse model \grad q_k(y)
    //           metric for \grad q_k(y) can differ from gradient of w and <w, .>_y
    // TODO: move computation of w to CoarseOracleBase to avoid mismatches in metric between w and <w,.>
    //       const correctness for CoarseOracleBase
    CoarseModel(const GrossPitaevskiiOracle<dim>& O_coarse, const GrossPitaevskiiOracle<dim>& O_fine,
                GrossPitaevskiiCoarseOracle<dim>& qk,
                const ManifoldTransferBase& point_transfer,
                const VectorTransportBase& vector_transport)
        : O_coarse(O_coarse), O_fine(O_fine)
        , qk(qk)
        , point_transfer(point_transfer)
        , vector_transport(vector_transport)
        , n_coarse(O_coarse.n_dofs())
        , n_fine(O_fine.n_dofs())
    {
        AssertThrow(O_coarse.get_metric() == O_fine.get_metric(), dealii::ExcInternalError("mismatch between fine and coarse oracle"));
    }

    void set_timer(const dealii::Timer& timer_new) const
    {
        timer = timer_new;
    }

    // We separate the "setup" phase (fine/coarse vectors) from the "model" phase
    // so that a coarse criterion can be efficiently evaluated.
    // TODO: save CoarseStep inside of class (with const ref accessor), instead of returning?
    CoarseStep
    setup(const Vector<double>& x) const
    {
        AssertDimension(x.size(),n_fine);
        CoarseStep step(x, n_coarse, n_fine);

        // Starting coarse point
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: point transfer\n";
#endif
        point_transfer.restriction(x, step.y);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: assemble matrix\n";
#endif
        O_coarse.update(step.y);  // mutable state (non-linear factor Mpp)

        // Compute coarse M-gradient (-> coarse correction w)
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_coarse.id << "-coarse gradient\n";
#endif
        O_coarse.gradient(step.y, step.y_grad);

        // Compute fine M-gradient (-> coarse correction w)
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << O_fine.id << "-fine gradient\n";
#endif
        O_fine.gradient(x, step.x_grad);

        // Compute restricted gradient (-> coarse correction w)
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: M-vector restriction\n";
#endif
        vector_transport.vector_restriction(step.y, x, step.x_grad, step.x_grad_restr);

        return step;
    }

    // TODO: Use cycle_fine() for consistency instead of gradient_descent()
    void solve(const CoarseStep& step, DescentOptions options_gd, Vector<double>& dst)
    {
        AssertDimension(dst.size(), n_fine);
        const auto& M_coarse = O_coarse.get_M();

        Vector<double> w(n_coarse);
        w = step.y_grad;
        w.add(-1.0, step.x_grad_restr);

        // Set up coarse model
        // TODO: const correctness - view to OracleBase
        qk.update_parameters(w, step.y);
        Vector<double> zk(n_coarse);

        // Find zk such that qk(zk) < qk(x)
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << qk.id << "-gradient descent\n";
#endif
        zk = gradient_descent(qk, step.y, options_gd, std::cerr);

        // Compute the search direction, zk <- L_x(zk)
        // FIXME? use separate variable to hold ambient tangent vector L_x(zk)
#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: inverse retraction\n";
#endif
        ellipsoid::retract_inv_by_norm(M_coarse, zk, step.y);

#ifdef CPU_TIME
        std::cerr << "[" << timer.cpu_time() << "] coarse: " << vector_transport.id << "-vector prolongation\n";
#endif
        vector_transport.vector_prolongation(step.x, step.y, zk, dst);
    }

    // Compute norms for coarse condition
    // TODO: the norm on the fine level is assumed to be consistent with the norm the coarse level
    //       if CoarseOracle(Base) supports two levels of discretization, this can be moved there
    CoarseCond
    norm(const Vector<double>& v, const Vector<double>& v_restr) const
    {
        AssertDimension(v.size(),n_fine);
        AssertDimension(v_restr.size(),n_coarse);
        CoarseCond cond;

        if (qk.get_metric() == MetricKind::MASS) {
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: M-norm of gradient\n";
#endif
            const auto& M_fine = O_fine.get_M();

            Vector<double> Mv(n_fine);
            M_fine.vmult(Mv, v);
            cond.norm_fine   = std::sqrt(v*Mv);

#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] coarse: M-norm of restricted gradient\n";
#endif
            const auto& M_coarse = O_coarse.get_M();

            Vector<double> Mg_restr(n_coarse);
            M_coarse.vmult(Mg_restr, v_restr);
            cond.norm_coarse = std::sqrt(v_restr*Mg_restr);
        }
        else if (qk.get_metric() == MetricKind::FROBENIUS) {
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: F-norm of gradient\n";
#endif
            cond.norm_fine   = std::sqrt(v*v);

#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] coarse: F-norm of restricted gradient\n";
#endif
            cond.norm_coarse = std::sqrt(v_restr*v_restr);
        }
        else {
            throw dealii::ExcNotImplemented("unknown metric for coarse model");
        }

        return cond;
    }

    unsigned n_dofs() const { return n_coarse; }

private:
    mutable dealii::Timer timer;

    // Function evaluation
    const GrossPitaevskiiOracle<dim>& O_coarse;  // coarse objective
    const GrossPitaevskiiOracle<dim>& O_fine;    // fine objective
    // TODO: const correctness?
    GrossPitaevskiiCoarseOracle<dim>& qk;  // coarse objective + linear shift

    // Grid operators
    const ManifoldTransferBase& point_transfer;
    const VectorTransportBase& vector_transport;

    // Degrees of freedom
    unsigned n_coarse;
    unsigned n_fine;
};


template <int dim>
class GradientDescent
{
public:
    // O_fine: oracle used for computing gradient descent steps on the fine level
    GradientDescent(const GrossPitaevskiiOracle<dim>& O_fine)
        : O_fine(O_fine)
    {}

    void cycle(const Vector<double>& x0, std::ostream& os, DescentOptions options_gd)
    {
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> x(x0);
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Update gradient
#ifdef CPU_TIME
            std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
#endif
            auto lac_iter = O_fine.gradient(x, x_grad);
            dk  = x_grad;
            dk *= -1.0;

            // Evaluate directional derivative in A-norm
            CycleInfo info = cycle_smooth(O_fine, x, dk, timer, options_gd);
            info.iter      = i;
            info.coarse    = false;
            info.lac_iter  = lac_iter;

            cycle_eval(O_fine, x, convergence_table, info);
        }
        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;
    const GrossPitaevskiiOracle<dim>& O_fine;
};


// TODO: convergence check (cf. EnergySimulator::run)
template <int dim>
class FullApproximationScheme
{
public:
    // TODO: to support recursion, we need to be able to take arbitrary Oracles at levels 0...{n-1}
    // O_fine: oracle used for computing gradient descent steps on the fine level
    //         independent from oracles used in CoarseModel (w, <w,.> and corresponding objectives)
    // O_coarse_model:
    //         model used for computing coarse descent steps
    // TODO: const correctness for CoarseModel
    FullApproximationScheme(const GrossPitaevskiiOracle<dim>& O_fine, CoarseModel<dim>& O_coarse_model)
        : O_fine(O_fine)
        , O_coarse_model(O_coarse_model)
    {
        O_coarse_model.set_timer(timer);
    }

    // TODO: move options to constructor for common cycle() interface?
    void cycle(const Vector<double>& x0, std::ostream& os,
               DescentOptions options_gd, DescentOptions options_gd_coarse,
               double kappa, double eps, unsigned coarse_every = 1)
    {
        timer.reset();

        // 0. Initialize oracle and coarse model
        Vector<double> x(x0);
        O_fine.update(x);
        Vector<double> x_grad(x.size());
        Vector<double> dk(x.size());

        // FIXME: abstraction leak - implement coarse criterion in CoarseModel
        bool check_coarse_cond = true;

        for (unsigned i = 0; i < options_gd.max_iter; i++) {
            // Compute coarse condition
            if (check_coarse_cond && (i == 0 || i % coarse_every == 0)) {
                // TODO: If the coarse model is evaluated in the A-gradient (or the fine objected solved
                //       in the M-metric), this step results in negligible additional effort.
                //       Metric-free formulation of the coarse condition?
                auto coarse_step = O_coarse_model.setup(x);
                auto coarse_cond = O_coarse_model.norm(coarse_step.x_grad, coarse_step.x_grad_restr);

                // Norm of fine (M-)gradient
                convergence_table.add_value("grad_norm", coarse_cond.norm_fine);
                // Norm of restricted (M-)gradient
                convergence_table.add_value("grad_restr_norm", coarse_cond.norm_coarse);

                if (coarse_cond.norm_coarse <= eps) {
                    check_coarse_cond = false;  // stop coarse condition evaluation once threshold was reached
                }

                if (coarse_cond.norm_coarse >= kappa*coarse_cond.norm_fine && coarse_cond.norm_coarse > eps) {
                    // Coarse step
                    O_coarse_model.solve(coarse_step, options_gd_coarse, dk);

                    // TODO: debug step for checking descent direction (to gradient of corresponding fine oracle)
                    CycleInfo info = cycle_smooth(O_fine, x, dk, timer, options_gd);
                    info.iter      = i;
                    info.coarse    = true;
                    info.lac_iter  = 0;

                    cycle_eval(O_fine, x, convergence_table, info);
                } else {
                    goto fine_step;
                }
            } else {
                convergence_table.add_value("grad_restr_norm", 0);
                convergence_table.add_value("grad_norm", 0);
    fine_step:
                // Update gradient
#ifdef CPU_TIME
                std::cerr << "[" << timer.cpu_time() << "] fine: A-gradient\n";
#endif
                auto lac_iter = O_fine.gradient(x, x_grad);
                dk  = x_grad;
                dk *= -1.0;

                // Evaluate directional derivative in A-norm
                CycleInfo info = cycle_smooth(O_fine, x, dk, timer, options_gd);
                info.iter      = i;
                info.coarse    = false;
                info.lac_iter  = lac_iter;

                cycle_eval(O_fine, x, convergence_table, info);
            }
        }
        convergence_table.set_precision("grad_restr_norm", 4);
        convergence_table.set_precision("grad_norm", 4);
        convergence_table.set_scientific("grad_restr_norm", true);
        convergence_table.set_scientific("grad_norm", true);

        cycle_finalize(convergence_table, os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

private:
    dealii::ConvergenceTable convergence_table;
    dealii::Timer timer;
    const GrossPitaevskiiOracle<dim>& O_fine;
    // TODO: const correctness (update_parameters) - compute tilt inside CoarseOracleBase
    CoarseModel<dim>& O_coarse_model;  // encodes both the coarse model, and the method to solve it
};

} // namespace gpe

#endif //GPE_MAIN_COARSE_H
