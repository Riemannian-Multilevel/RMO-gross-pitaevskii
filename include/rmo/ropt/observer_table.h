#ifndef RMO_ROPT_CONVERGENCE_OBSERVER_H
#define RMO_ROPT_CONVERGENCE_OBSERVER_H

#include <rmo/ropt/observer.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>

#include <ostream>
#include <string>

namespace rmo
{

struct ConvergenceTable : dealii::ConvergenceTable {
    bool has_column(const std::string& key) const {
        return columns.count(key) > 0;
    }
};


//! Records the iteration history in one table per level, written when that level's cycle finishes.
class ConvergenceTableObserver : public IterationObserver
{
public:
    explicit ConvergenceTableObserver(unsigned min_level = 0, unsigned max_level = 0)
    {
        conv_table_mg.resize(min_level, max_level);
    }

    void begin_level(unsigned level) override { conv_table_mg[level].clear(); }

    void add(const CycleInfo& info) override
    {
        auto& table = conv_table_mg[info.level];

        // Added first so they precede the common columns
        if (info.coarse_cond) {
            table.add_value("grad_norm",       info.grad_norm);
            table.add_value("grad_restr_norm", info.grad_restr_norm);
        }
        table.add_value("iter",     info.iter);
        table.add_value("level",    info.level);
        table.add_value("coarse",   info.coarse ? "*" : " ");
        table.add_value("lac_iter", info.lac_iter);
        table.add_value("residual", info.residual);
        table.add_value("energy",   info.energy);
        table.add_value("step",     info.step_size);
        table.add_value("elapsed",  info.elapsed);
    }

    void end_level(unsigned level, std::ostream& os) override
    {
        auto& table = conv_table_mg[level];

        // grad_* are absent on the coarsest level and in single-level descent
        for (const char* col : {"residual", "step", "elapsed", "grad_norm", "grad_restr_norm"}) {
            if (table.has_column(col)) {
                table.set_precision(col, 4);
                table.set_scientific(col, true);
            }
        }
        table.set_precision("energy", 16);
        table.set_scientific("energy", true);

        table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate);
        table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate_log2);
        table.write_text(os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

    const ConvergenceTable& get_table(unsigned level) const { return conv_table_mg[level]; }

private:
    dealii::MGLevelObject<ConvergenceTable> conv_table_mg;
};

} // namespace rmo

#endif //RMO_ROPT_CONVERGENCE_OBSERVER_H
