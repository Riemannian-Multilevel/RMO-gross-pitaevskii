//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_RAWIO_H
#define RMO_CC_RAWIO_H

#include <rmo/lac.h>

#include <fstream>
#include <stdexcept>
#include <string>

namespace rmo::cc
{

/**
 * @brief Loads a raw, row-major, native-byte-order float64 raster with no header --
 * the format @c test/cc/prepare_cow_data.py writes, one array per file, dimensions
 * encoded only in the filename (the caller already knows them, as for @ref SyntheticImage).
 *
 * @param n Expected element count (rows*cols); a short or long file is treated as an error
 * rather than silently truncated or zero-padded, since a size mismatch here means the
 * caller's (rows, cols) do not match the file that was requested.
 */
inline Vector<double> load_raster(const std::string& path, unsigned int n)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cc::load_raster: cannot open " + path);
    }

    Vector<double> v(n);
    const auto n_bytes = static_cast<std::streamsize>(n) * static_cast<std::streamsize>(sizeof(double));
    in.read(reinterpret_cast<char*>(v.begin()), n_bytes);
    if (in.gcount() != n_bytes) {
        throw std::runtime_error("cc::load_raster: " + path + " does not hold " +
                                 std::to_string(n) + " float64 values");
    }
    // Reject a longer-than-expected file too (in.peek() after a full read would already be
    // EOF for an exact match).
    in.peek();
    if (!in.eof()) {
        throw std::runtime_error("cc::load_raster: " + path + " holds more than " +
                                 std::to_string(n) + " float64 values");
    }
    return v;
}

} // namespace rmo::cc

#endif //RMO_CC_RAWIO_H
