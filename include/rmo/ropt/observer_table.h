#ifndef RMO_ROPT_CONVERGENCE_OBSERVER_H
#define RMO_ROPT_CONVERGENCE_OBSERVER_H

#include <rmo/ropt/observer.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>

#include <ostream>
#include <set>
#include <string>
#include <type_traits>

/**
 * @file
 * @brief ConvergenceTableObserver: iteration history as one convergence table per level.
 */
namespace rmo
{

struct ConvergenceTable : dealii::ConvergenceTable {
    bool has_column(const std::string& key) const {
        return columns.count(key) > 0;
    }
};


//! Records the iteration history in one table per level, written when that level's cycle finishes.
//! InfoType: described with BOOST_DESCRIBE_STRUCT (one column per described member) and has a member
//! `level`; an optional member `extra` (pairs of name and double) adds further columns.
template <typename InfoType>
class ConvergenceTableObserver : public IterationObserver<InfoType>
{
public:
    explicit ConvergenceTableObserver(unsigned min_level = 0, unsigned max_level = 0)
    {
        conv_table_mg.resize(min_level, max_level);
    }

    void begin_level(unsigned level) override { conv_table_mg[level].clear(); }

    void add(const InfoType& info) override
    {
        auto& table = conv_table_mg[info.level];

        // Added first so they precede the common columns
        if constexpr (requires { info.extra; }) {
            for (const auto& [name, value] : info.extra) {
                add_float(table, name, value);
            }
        }
        boost::mp11::mp_for_each<boost::describe::describe_members<InfoType, boost::describe::mod_public>>([&](auto D) {
            auto value = info.*D.pointer;

            if constexpr (std::is_same_v<decltype(value), bool>) {
                table.add_value(D.name, value ? "*" : " ");
            }
            else if constexpr (std::is_floating_point_v<decltype(value)>) {
                add_float(table, D.name, value);
            }
            else {
                table.add_value(D.name, value);
            }
        });
    }

    void end_level(unsigned level, std::ostream& os) override
    {
        auto& table = conv_table_mg[level];

        for (const auto& col : m_float_columns) {
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
    void add_float(ConvergenceTable& table, const std::string& name, double value)
    {
        table.add_value(name, value);
        m_float_columns.insert(name);
    }

    dealii::MGLevelObject<ConvergenceTable> conv_table_mg;
    std::set<std::string> m_float_columns;  ///< Columns written in scientific notation
};

} // namespace rmo

#endif //RMO_ROPT_CONVERGENCE_OBSERVER_H
