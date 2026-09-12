//
// Created by Ferdinand Vanmaele.
//

#ifndef RMO_CC_SYNTHETIC_H
#define RMO_CC_SYNTHETIC_H

#include <rmo/lac.h>

#include <cmath>

namespace rmo::cc
{

/**
 * @brief A bright disk on a dark background, with a foreground/background seed patch known
 * to lie inside/outside it -- a stand-in for a loaded image and its user-provided seed
 * regions (cf. Fig. 12), used by the drivers so they need no image I/O.
 */
struct SyntheticImage
{
    unsigned int rows, cols;
    Vector<double> raster;         // grayscale intensity in [0,1], row-major
    unsigned int patch = 5;        // seed patch side length
    unsigned int fg_row, fg_col;   // foreground seed patch (top-left corner)
    unsigned int bg_row, bg_col;   // background seed patch (top-left corner)

    explicit SyntheticImage(unsigned int n)
        : rows(n), cols(n), raster(n * n)
        , fg_row(n / 2 - patch / 2), fg_col(n / 2 - patch / 2)
        , bg_row(0), bg_col(0)
    {
        const double cy = (rows - 1) / 2.0, cx = (cols - 1) / 2.0;
        const double radius = std::min(rows, cols) / 4.0;

        for (unsigned int r = 0; r < rows; r++) {
            for (unsigned int c = 0; c < cols; c++) {
                const double dr = r - cy, dc = c - cx;
                raster[r * cols + c] = (std::sqrt(dr * dr + dc * dc) <= radius) ? 0.8 : 0.2;
            }
        }
    }

    [[nodiscard]] double patch_mean(unsigned int row0, unsigned int col0) const
    {
        double s = 0.0;
        for (unsigned int r = row0; r < row0 + patch; r++) {
            for (unsigned int c = col0; c < col0 + patch; c++) {
                s += raster[r * cols + c];
            }
        }
        return s / (patch * patch);
    }
};

/** @brief Data term @f$ \rho_i = (c_f - g_i)^2 - (c_b - g_i)^2 @f$ (eq. (41)/(42)), in raster order. */
inline Vector<double> build_rho(const SyntheticImage& img)
{
    const double c_f = img.patch_mean(img.fg_row, img.fg_col);
    const double c_b = img.patch_mean(img.bg_row, img.bg_col);

    Vector<double> rho(img.raster.size());
    for (unsigned int i = 0; i < rho.size(); i++) {
        const double g = img.raster[i];
        rho[i] = (c_f - g) * (c_f - g) - (c_b - g) * (c_b - g);
    }
    return rho;
}

} // namespace rmo::cc

#endif //RMO_CC_SYNTHETIC_H
