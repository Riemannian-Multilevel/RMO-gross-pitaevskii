#ifndef RMO_TEST_CHECK_H
#define RMO_TEST_CHECK_H

#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

/**
 * @file
 * @brief Test harness shared by the tests: CheckReport prints one PASS/FAIL line per check and counts the
 * failures, run_tests() turns the report and uncaught exceptions into the exit code.
 */
namespace rmo::test
{

//! Exit code that ctest reports as a skipped test (SKIP_RETURN_CODE).
constexpr int skip_code = 77;

//! @p x in scientific notation with 2 digits, for the details of a check.
inline std::string sci(double x)
{
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << x;
    return ss.str();
}

//! Prints one line per check and counts the failures.
class CheckReport
{
public:
    explicit CheckReport(std::ostream& out = std::cout) : out(out) {}

    //! Prints "PASS  name  (detail)" or "FAIL  name  (detail)" and returns @p ok.
    bool check(bool ok, std::string_view name, std::string_view detail = {})
    {
        print(ok ? "PASS  " : "FAIL  ", name, detail);
        n_failed += !ok;
        return ok;
    }

    //! Prints "INFO  name  (detail)": a reported value that is not checked.
    void info(std::string_view name, std::string_view detail = {})
    {
        print("INFO  ", name, detail);
    }

    //! Prints the summary and returns the exit code: 0 if all checks passed, 1 otherwise.
    int finish() const
    {
        out << (n_failed ? std::to_string(n_failed) + " check(s) failed" : "all checks passed") << std::endl;
        return n_failed ? 1 : 0;
    }

private:
    void print(std::string_view tag, std::string_view name, std::string_view detail)
    {
        out << tag << name;
        if (!detail.empty()) {
            out << "  (" << detail << ")";
        }
        out << "\n";
    }

    std::ostream& out;
    unsigned n_failed = 0;
};

//! Runs @p tests on a CheckReport and returns the exit code; an uncaught exception is a failed check.
template <typename Tests>
int run_tests(Tests&& tests)
{
    CheckReport report;
    try {
        tests(report);
    }
    catch (const std::exception& e) {
        report.check(false, "uncaught exception", e.what());
    }
    return report.finish();
}

} // namespace rmo::test

#endif // RMO_TEST_CHECK_H
