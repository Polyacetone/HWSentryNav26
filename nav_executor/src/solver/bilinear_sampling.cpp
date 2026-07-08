#include <nav_executor/solver/bilinear_sampling.hpp>
#include <cmath>

namespace nav_executor {

CostSample eval_cost_bilinear(const CostMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    CostSample s {255.0, 0.0, 0.0};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 2 || info.height < 2) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(info.width - 1)
        || gy >= static_cast<double>(info.height - 1)) {
        return s;
    }

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(ix0);
    const double ty = gy - static_cast<double>(iy0);

    const double f00 = grid.value_at_clamped(iy0, ix0);
    const double f10 = grid.value_at_clamped(iy0, ix0 + 1);
    const double f01 = grid.value_at_clamped(iy0 + 1, ix0);
    const double f11 = grid.value_at_clamped(iy0 + 1, ix0 + 1);

    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    s.value = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;

    const double dvdgx = (1.0 - ty) * (f10 - f00) + ty * (f11 - f01);
    const double dvdgy = (1.0 - tx) * (f01 - f00) + tx * (f11 - f10);
    s.dx = dvdgx * info.inv_resolution;
    s.dy = dvdgy * info.inv_resolution;
    return s;
}

DirSample eval_dir_bilinear(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    DirSample s {Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()};
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return s;
    if (info.width < 2 || info.height < 2) return s;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(info.width - 1)
        || gy >= static_cast<double>(info.height - 1)) {
        return s;
    }

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(ix0);
    const double ty = gy - static_cast<double>(iy0);

    const Eigen::Vector2d f00 = grid.value_at_clamped(iy0, ix0);
    const Eigen::Vector2d f10 = grid.value_at_clamped(iy0, ix0 + 1);
    const Eigen::Vector2d f01 = grid.value_at_clamped(iy0 + 1, ix0);
    const Eigen::Vector2d f11 = grid.value_at_clamped(iy0 + 1, ix0 + 1);

    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    s.value = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;

    const Eigen::Vector2d dvdgx = (1.0 - ty) * (f10 - f00) + ty * (f11 - f01);
    const Eigen::Vector2d dvdgy = (1.0 - tx) * (f01 - f00) + tx * (f11 - f10);
    s.J.col(0) = dvdgx * info.inv_resolution;
    s.J.col(1) = dvdgy * info.inv_resolution;
    return s;
}

Eigen::Vector2d eval_dir_bilinear_value_only(const DirectionMapGridView& grid, const GridInfo& info, double x_map, double y_map) {
    Eigen::Vector2d val = Eigen::Vector2d::Zero();
    if (!std::isfinite(x_map) || !std::isfinite(y_map)) return val;
    if (info.width < 2 || info.height < 2) return val;

    const double gx = (x_map - info.origin_x) * info.inv_resolution;
    const double gy = (y_map - info.origin_y) * info.inv_resolution;
    if (gx < 0.0 || gy < 0.0 || gx >= static_cast<double>(info.width - 1)
        || gy >= static_cast<double>(info.height - 1)) {
        return val;
    }

    const int ix0 = static_cast<int>(std::floor(gx));
    const int iy0 = static_cast<int>(std::floor(gy));
    const double tx = gx - static_cast<double>(ix0);
    const double ty = gy - static_cast<double>(iy0);

    const Eigen::Vector2d f00 = grid.value_at_clamped(iy0, ix0);
    const Eigen::Vector2d f10 = grid.value_at_clamped(iy0, ix0 + 1);
    const Eigen::Vector2d f01 = grid.value_at_clamped(iy0 + 1, ix0);
    const Eigen::Vector2d f11 = grid.value_at_clamped(iy0 + 1, ix0 + 1);

    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;

    return w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;
}

} // namespace nav_executor
