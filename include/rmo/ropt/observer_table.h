#ifndef RMO_ROPT_CONVERGENCE_OBSERVER_H
#define RMO_ROPT_CONVERGENCE_OBSERVER_H

#include <rmo/ropt/observer.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>

#include <boost/describe.hpp>
#include <boost/mp11.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>

/**
 * @file
 * @brief Observers writing the iteration history as tables: SimpleTableObserver (one continuous table, rows
 * written as they arrive) and ConvergenceTableObserver (one table per level, written when its cycle finishes).
 *
 * InfoType is described with BOOST_DESCRIBE_STRUCT: one column per described member, in the order of the
 * description. An unset `std::optional` member is a blank cell (SimpleTableObserver) or adds no value
 * (ConvergenceTableObserver). Floating-point columns are written in scientific notation with the digits given
 * by ColumnPrecision.
 */
namespace rmo
{

namespace detail
{
template <typename T> inline constexpr bool is_optional_v = false;
template <typename T> inline constexpr bool is_optional_v<std::optional<T>> = true;
} // namespace detail


//! Digits of the floating-point columns: `digits`, or the override of the column.
struct ColumnPrecision
{
    unsigned digits = 4;
    std::map<std::string, unsigned, std::less<>> overrides = {{"energy", 16}};

    unsigned operator()(std::string_view column) const
    {
        const auto it = overrides.find(column);
        return it != overrides.end() ? it->second : digits;
    }
};

/**
 * @brief SimpleTableObserver: iteration history as one continuous org-mode table.
 *
 * The header is written on construction, each row in add(); rows of all levels form one table (column
 * `level`). Every described member is a column, so the number of columns is fixed: an unset optional is
 * written as a blank cell. Unlike ConvergenceTableObserver, there are no reduction rates, and
 * begin_level()/end_level() do nothing.
 */
template <typename InfoType>
class SimpleTableObserver : public IterationObserver<InfoType>
{
    using Members = boost::describe::describe_members<InfoType, boost::describe::mod_public>;

public:
    explicit SimpleTableObserver(std::ostream& os, ColumnPrecision precision = {})
        : m_os(os)
        , m_precision(std::move(precision))
    {
        write_row([](auto D) { return std::string(D.name); });
    }

    void add(const InfoType& info) override
    {
        write_row([&](auto D) { return cell(info.*D.pointer, D.name); });
    }

private:
    //! Writes one row with the cells `cell_of(D)` of the described members D, and flushes.
    template <typename CellOf>
    void write_row(CellOf&& cell_of)
    {
        std::string line = "|";
        boost::mp11::mp_for_each<Members>([&](auto D) {
            // Column width: the name, or a floating-point value with sign (-1.2345e-08)
            const size_t width = std::max<size_t>(std::strlen(D.name), m_precision(D.name) + 7);
            line += fmt::format(" {:>{}} |", cell_of(D), width);
        });
        m_os << line << std::endl;   // flush: rows appear while the solver runs
    }

    template <typename T>
    std::string cell(const T& value, std::string_view name) const
    {
        if constexpr (detail::is_optional_v<T>) {
            return value ? cell(*value, name) : std::string();
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return value ? "*" : "";
        }
        else if constexpr (std::is_floating_point_v<T>) {
            return fmt::format("{:.{}e}", value, m_precision(name));
        }
        else {
            return fmt::format("{}", value);
        }
    }

    std::ostream&   m_os;         ///< Output stream for the header and the rows
    ColumnPrecision m_precision;
};


struct ConvergenceTable : dealii::ConvergenceTable {
    bool has_column(const std::string& key) const {
        return columns.count(key) > 0;
    }
};


/**
 * @brief ConvergenceTableObserver: iteration history as one convergence table per level.
 *
 * Records the iteration history in one table per level, written when that level's cycle finishes.
 * InfoType: described with BOOST_DESCRIBE_STRUCT (one column per described member) and has a member `level`;
 * an unset `std::optional` member adds no value, so its column exists only on levels where it is set.
 */
template <typename InfoType>
class ConvergenceTableObserver : public IterationObserver<InfoType>
{
public:
    explicit ConvergenceTableObserver(unsigned min_level = 0, unsigned max_level = 0, ColumnPrecision precision = {})
        : m_precision(std::move(precision))
    {
        conv_table_mg.resize(min_level, max_level);
    }

    void begin_level(unsigned level) override { conv_table_mg[level].clear(); }

    void add(const InfoType& info) override
    {
        auto& table = conv_table_mg[info.level];

        boost::mp11::mp_for_each<boost::describe::describe_members<InfoType, boost::describe::mod_public>>([&](auto D) {
            add_cell(table, D.name, info.*D.pointer);
        });
    }

    void end_level(unsigned level, std::ostream& os) override
    {
        auto& table = conv_table_mg[level];

        for (const auto& col : m_float_columns) {
            if (table.has_column(col)) {
                table.set_precision(col, m_precision(col));
                table.set_scientific(col, true);
            }
        }

        table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate);
        table.evaluate_convergence_rates("residual", dealii::ConvergenceTable::reduction_rate_log2);
        table.write_text(os, dealii::TableHandler::TextOutputFormat::org_mode_table);
    }

    const ConvergenceTable& get_table(unsigned level) const { return conv_table_mg[level]; }

private:
    //! Adds @p value to the column @p name; an unset optional adds nothing.
    template <typename T>
    void add_cell(ConvergenceTable& table, const std::string& name, const T& value)
    {
        if constexpr (detail::is_optional_v<T>) {
            if (value) {
                add_cell(table, name, *value);
            }
        }
        else if constexpr (std::is_same_v<T, bool>) {
            table.add_value(name, value ? "*" : " ");
        }
        else if constexpr (std::is_floating_point_v<T>) {
            table.add_value<double>(name, value);
            m_float_columns.insert(name);
        }
        else {
            table.add_value(name, value);
        }
    }

    ColumnPrecision m_precision;
    dealii::MGLevelObject<ConvergenceTable> conv_table_mg;
    std::set<std::string> m_float_columns;  ///< Columns written in scientific notation
};


//! SimpleTableObserver writing to @p os if @p simple, otherwise a ConvergenceTableObserver for the levels
//! @p min_level to @p max_level (which writes to the stream passed by the solver).
template <typename InfoType>
std::unique_ptr<IterationObserver<InfoType>>
make_table_observer(bool simple, std::ostream& os, unsigned min_level = 0, unsigned max_level = 0)
{
    if (simple) {
        return std::make_unique<SimpleTableObserver<InfoType>>(os);
    }
    return std::make_unique<ConvergenceTableObserver<InfoType>>(min_level, max_level);
}

} // namespace rmo

#endif //RMO_ROPT_CONVERGENCE_OBSERVER_H
