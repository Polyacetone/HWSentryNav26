#include <nav_executor/path_planner/search/route_terrain_mask.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nav_executor {

RouteTerrainMask::RouteTerrainMask(const RouteTerrainMaskParams& params) : params_(params) {}

void RouteTerrainMask::initialize(const CostMap& cost_map, DirectionMap::ConstPtr direction_map) {
    if (!direction_map) {
        throw std::invalid_argument("RouteTerrainMask::initialize: direction_map is null");
    }
    if (!cost_map.geometry.same_geometry(direction_map->geometry)) {
        throw std::runtime_error("RouteTerrainMask::initialize: direction_map geometry mismatch");
    }

    base_direction_map_ = std::move(direction_map);

    std::vector<uint8_t> base_terrain_cost_data(base_direction_map_->data.size());
    for (size_t idx = 0; idx < base_direction_map_->data.size(); ++idx) {
        const double mag = base_direction_map_->data[idx].norm();
        base_terrain_cost_data[idx] = static_cast<uint8_t>(
            std::clamp(mag * 255.0, 0.0, 255.0)
        );
    }
    base_terrain_cost_layer_ = std::make_shared<const CostMap>(
        base_direction_map_->geometry, std::move(base_terrain_cost_data)
    );

    build_kernel(base_direction_map_->geometry.resolution());
}

RouteTerrainMask::Layers RouteTerrainMask::compute(const std::optional<MincoTrajectory>& global_path) const {
    if (!base_direction_map_) return {};

    std::vector<double> max_alpha(base_terrain_cost_layer_->data.size(), 0.0);

    if (global_path && !kernel_.empty()) {
        const auto& trajectory = *global_path;
        const double length = trajectory.total_arc_length();
        if (length > 0.0) {
            const double sample_spacing = base_direction_map_->geometry.resolution() * 0.5;
            const int samples = std::max(1, static_cast<int>(std::ceil(length / sample_spacing)));

            for (int i = 0; i <= samples; ++i) {
                const double arc_length = length * static_cast<double>(i)
                    / static_cast<double>(samples);
                const TrajSample sample = trajectory.eval_arc_length(arc_length);
                const Eigen::Vector2d& pos = sample.p;
                Eigen::Vector2d tangent = sample.dp_dtau;
                const double tangent_norm = tangent.norm();
                if (tangent_norm < 1e-6) continue;
                tangent /= tangent_norm;

                const auto erase_center = base_direction_map_->geometry.containing_cell(pos);
                if (!erase_center
                    || !base_direction_map_->is_terrain_body_cell(*erase_center)) {
                    continue;
                }

                Eigen::Vector2d direction = base_direction_map_->raw_direction_at_cell(
                    *erase_center
                );
                const double direction_norm = direction.norm();
                if (direction_norm < 1e-6) continue;
                direction /= direction_norm;
                const double alignment_cosine = std::abs(tangent.dot(direction));
                if (alignment_cosine < params_.min_alignment_cosine) continue;

                apply_kernel_at(*erase_center, max_alpha);
            }
        }
    }

    // 路径附近降低方向地形代价，使本路线选择的穿越通道保持可通行。
    std::vector<uint8_t> step_cost_data = base_terrain_cost_layer_->data;
    for (size_t idx = 0; idx < step_cost_data.size(); ++idx) {
        const double a = std::clamp(max_alpha[idx], 0.0, 1.0);
        const double base = static_cast<double>(base_terrain_cost_layer_->data[idx]);
        const double v = base * (1.0 - a);
        step_cost_data[idx] = static_cast<uint8_t>(std::clamp(std::lround(v), 0l, 255l));
    }

    auto step_cost_layer = std::make_shared<const CostMap>(
        base_direction_map_->geometry, std::move(step_cost_data)
    );

    // 仅保留路径附近的方向场，避免控制器误用其他方向地形约束。
    std::vector<Eigen::Vector2d> masked_dir_data;
    masked_dir_data.reserve(base_direction_map_->data.size());
    for (size_t idx = 0; idx < base_direction_map_->data.size(); ++idx) {
        const double a = std::clamp(max_alpha[idx], 0.0, 1.0);
        masked_dir_data.push_back(base_direction_map_->data[idx] * a);
    }

    auto masked_direction_map = std::make_shared<const DirectionMap>(
        base_direction_map_->geometry,
        std::move(masked_dir_data),
        base_direction_map_->terrain
    );

    return {std::move(step_cost_layer), std::move(masked_direction_map)};
}

void RouteTerrainMask::build_kernel(double resolution) {
    kernel_.clear();
    const int cutoff_radius = static_cast<int>(std::ceil(params_.cutoff_radius / resolution));

    kernel_.reserve(static_cast<size_t>((2 * cutoff_radius + 1) * (2 * cutoff_radius + 1)));
    for (int dy = -cutoff_radius; dy <= cutoff_radius; ++dy) {
        for (int dx = -cutoff_radius; dx <= cutoff_radius; ++dx) {
            const double dist = resolution
                * std::hypot(static_cast<double>(dx), static_cast<double>(dy));
            if (dist > params_.cutoff_radius) continue;

            const double alpha = dist <= params_.full_effect_radius ? 1.0
                : (1.0 - (dist - params_.full_effect_radius)
                    / (params_.cutoff_radius - params_.full_effect_radius));

            kernel_.push_back({dx, dy, std::clamp(alpha, 0.0, 1.0)});
        }
    }
}

void RouteTerrainMask::apply_kernel_at(const Eigen::Vector2i& grid_coord, std::vector<double>& max_alpha) const {
    if (!base_direction_map_) return;
    if (kernel_.empty()) return;

    const int width = base_direction_map_->geometry.width();
    const int height = base_direction_map_->geometry.height();
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

} // namespace nav_executor
