#include <path_follower/step_routing_mask.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace path_follower {

StepRoutingMask::StepRoutingMask(const StepRoutingMaskParams& params) : params_(params) {}

void StepRoutingMask::initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map) {
    if (!direction_map) {
        throw std::invalid_argument("StepRoutingMask::initialize: direction_map is null");
    }
    if (cost_map.width != direction_map->width || cost_map.height != direction_map->height || cost_map.resolution != direction_map->resolution
        || cost_map.origin_x != direction_map->origin_x || cost_map.origin_y != direction_map->origin_y) {
        throw std::runtime_error("StepRoutingMask::initialize: direction_map geometry mismatch");
    }

    base_direction_map_ = std::move(direction_map);

    base_step_cost_data_.resize(base_direction_map_->data.size());
    for (size_t idx = 0; idx < base_direction_map_->data.size(); ++idx) {
        const double mag = base_direction_map_->data[idx].norm();
        base_step_cost_data_[idx] = static_cast<uint8_t>(std::clamp(mag * 255.0, 0.0, 255.0));
    }

    build_kernel(base_direction_map_->resolution);

    // 默认先生成一次“无路径经过”的输出层
    update(std::nullopt);
}

void StepRoutingMask::update(const std::optional<SplineD>& global_path) {
    if (!base_direction_map_) return;

    std::vector<double> max_alpha(base_step_cost_data_.size(), 0.0);

    if (global_path && !kernel_.empty()) {
        const auto& spline = *global_path;
        const double length = approximate_path_length(spline);
        if (length > 0.0) {
            const double sample_spacing = base_direction_map_->resolution * 0.5;
            const int samples = std::max(1, static_cast<int>(std::ceil(length / sample_spacing)));

            for (int i = 0; i <= samples; ++i) {
                const double u = static_cast<double>(i) / static_cast<double>(samples);
                const Eigen::Vector2d pos = spline.evaluate(u);

                Eigen::Vector2d tangent = spline.derivative(u, 1);
                const double tangent_norm = tangent.norm();
                if (tangent_norm < 1e-6) continue;
                tangent /= tangent_norm;

                const Eigen::Vector2d gc_dir = base_direction_map_->map_coord_to_grid(pos);
                if (!base_direction_map_->is_valid_coord(gc_dir)) continue;

                const Eigen::Vector2d dir = base_direction_map_->interpolate(gc_dir);
                const double dot = std::abs(tangent.dot(dir));
                if (dot <= params_.path_align_dot_threshold) continue;

                const Eigen::Vector2d gc_erase = gc_dir;
                const Eigen::Vector2i erase_center{
                    static_cast<int>(std::round(gc_erase.x())),
                    static_cast<int>(std::round(gc_erase.y()))
                };
                if (!base_direction_map_->is_valid_coord(erase_center)) continue;

                apply_kernel_at(erase_center, max_alpha);
            }
        }
    }

    // 1) 生成台阶额外代价层：乘法软擦除 cost <- cost*(1-alpha)
    std::vector<uint8_t> step_cost_data = base_step_cost_data_;
    for (size_t idx = 0; idx < step_cost_data.size(); ++idx) {
        const double a = std::clamp(max_alpha[idx], 0.0, 1.0);
        const double base = static_cast<double>(base_step_cost_data_[idx]);
        const double v = base * (1.0 - a);
        step_cost_data[idx] = static_cast<uint8_t>(std::clamp(std::lround(v), 0l, 255l));
    }

    step_cost_layer_ = std::make_shared<CostMap>(
        base_direction_map_->width,
        base_direction_map_->height,
        base_direction_map_->resolution,
        base_direction_map_->origin_x,
        base_direction_map_->origin_y,
        step_cost_data
    );

    // 2) 生成掩码后的方向场：乘法软保留 dir <- dir*alpha
    std::vector<Eigen::Vector2d> masked_dir_data;
    masked_dir_data.reserve(base_direction_map_->data.size());

    for (size_t idx = 0; idx < base_direction_map_->data.size(); ++idx) {
        const double a = std::clamp(max_alpha[idx], 0.0, 1.0);
        masked_dir_data.push_back(base_direction_map_->data[idx] * a);
    }

    masked_direction_map_ = std::make_shared<DirectionMap>(
        base_direction_map_->width,
        base_direction_map_->height,
        base_direction_map_->resolution,
        base_direction_map_->origin_x,
        base_direction_map_->origin_y,
        std::move(masked_dir_data)
    );
}

double StepRoutingMask::approximate_path_length(const SplineD& spline) const {
    const int samples = std::max(1, params_.length_num_samples);
    double len = 0.0;
    const double du = 1.0 / static_cast<double>(samples);
    for (int i = 0; i < samples; ++i) {
        const double u = (static_cast<double>(i) + 0.5) * du;
        const Eigen::Vector2d d1 = spline.derivative(u, 1);
        len += d1.norm() * du;
    }
    return len;
}

void StepRoutingMask::build_kernel(double resolution) {
    kernel_.clear();
    const int full_erase_radius = static_cast<int>(std::ceil(params_.full_effect_radius / resolution));
    const int cutoff_radius = static_cast<int>(std::ceil(params_.cutoff_radius / resolution));

    kernel_.reserve(static_cast<size_t>((2 * cutoff_radius + 1) * (2 * cutoff_radius + 1)));
    for (int dy = -cutoff_radius; dy <= cutoff_radius; ++dy) {
        for (int dx = -cutoff_radius; dx <= cutoff_radius; ++dx) {
            const double dist = std::hypot(static_cast<double>(dx), static_cast<double>(dy));
            if (dist > static_cast<double>(cutoff_radius)) continue;

            const double alpha = dist <= static_cast<double>(full_erase_radius) ? 1.0
                : (1.0 - (dist - static_cast<double>(full_erase_radius))
                    / (static_cast<double>(cutoff_radius) - static_cast<double>(full_erase_radius)));

            kernel_.push_back({dx, dy, std::clamp(alpha, 0.0, 1.0)});
        }
    }
}

void StepRoutingMask::apply_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<double>& max_alpha) const {
    if (!base_direction_map_) return;
    if (kernel_.empty()) return;

    const int width = base_direction_map_->width;
    const int height = base_direction_map_->height;
    const int cx = grid_coord.x();
    const int cy = grid_coord.y();

    for (const auto& cell : kernel_) {
        const int x = cx + cell.dx;
        const int y = cy + cell.dy;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        max_alpha[idx] = std::max(max_alpha[idx], cell.alpha);
    }
}

} // namespace path_follower
