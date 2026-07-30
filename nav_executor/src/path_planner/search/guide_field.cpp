#include <nav_executor/path_planner/search/guide_field.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>

#include <nav_executor/path_planner/search/grid_utils.hpp>

namespace nav_executor {

namespace {

const std::array<Eigen::Vector2i, 8> NEIGHBORS {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}};

struct OpenEntry {
    double distance = 0.0;
    Eigen::Vector2i cell = Eigen::Vector2i::Zero();

    bool operator>(const OpenEntry& other) const {
        return distance > other.distance;
    }
};

} // anonymous namespace

size_t GuideField::index(const Eigen::Vector2i& cell) const {
    return static_cast<size_t>(cell.y()) * static_cast<size_t>(geometry_->width())
        + static_cast<size_t>(cell.x());
}

const GuideFieldCell* GuideField::at_cell(const Eigen::Vector2i& cell) const {
    if (!geometry_ || !geometry_->contains_cell(cell)) return nullptr;
    const GuideFieldCell& value = cells_[index(cell)];
    return std::isfinite(value.distance_to_reference)
        && static_cast<double>(value.distance_to_reference) <= corridor_width_ + 1e-6
        ? &value : nullptr;
}

const GuideFieldCell* GuideField::at_map(const Eigen::Vector2d& point_map) const {
    if (!geometry_) return nullptr;
    const auto cell = geometry_->containing_cell(point_map);
    return cell ? at_cell(*cell) : nullptr;
}

bool GuideField::contains(const Eigen::Vector2i& cell) const {
    return at_cell(cell) != nullptr;
}

bool GuideField::contains_map(const Eigen::Vector2d& point_map) const {
    return at_map(point_map) != nullptr;
}

GuideFieldBuilder::Result GuideFieldBuilder::build(
    const ReferencePath& reference_path,
    const CostMap& cost_map,
    const int occupied_threshold
) const {
    Result result;
    if (reference_path.points.size() < 2 || params_.corridor_width <= 0.0
        || occupied_threshold <= 0) {
        result.error = "guide field configuration or reference path is invalid";
        return result;
    }
    GuideField& field = result.field;
    field.geometry_ = cost_map.geometry;
    field.corridor_width_ = params_.corridor_width;
    field.cells_.resize(
        static_cast<size_t>(cost_map.geometry.width())
            * static_cast<size_t>(cost_map.geometry.height())
    );
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, std::greater<>> open;

    const auto add_seed = [&](const Eigen::Vector2i& cell, const uint32_t reference_index) {
        if (!grid_cell_traversable(cost_map, cell, occupied_threshold)) return;
        GuideFieldCell& value = field.cells_[field.index(cell)];
        if (value.distance_to_reference == 0.0F
            && value.nearest_reference_index <= reference_index) {
            return;
        }
        value.distance_to_reference = 0.0F;
        value.nearest_reference_index = reference_index;
        open.push({0.0, cell});
    };

    for (size_t i = 0; i < reference_path.points.size(); ++i) {
        const auto cell = cost_map.geometry.containing_cell(
            reference_path.points[i].position
        );
        if (!cell) {
            result.error = "reference path seed leaves the guide field map";
            return result;
        }
        add_seed(*cell, static_cast<uint32_t>(i));
        if (i == 0) continue;
        for (const GridCrossing& crossing : trace_grid_crossings(
                cost_map.geometry,
                reference_path.points[i - 1].position,
                reference_path.points[i].position
            )) {
            const uint32_t nearest = crossing.fraction < 0.5
                ? static_cast<uint32_t>(i - 1) : static_cast<uint32_t>(i);
            add_seed(crossing.from, nearest);
            add_seed(crossing.to, nearest);
        }
    }
    if (open.empty()) {
        result.error = "guide field has no traversable reference seed";
        return result;
    }

    while (!open.empty()) {
        const OpenEntry current = open.top();
        open.pop();
        GuideFieldCell& current_value = field.cells_[field.index(current.cell)];
        if (current.distance > static_cast<double>(current_value.distance_to_reference)
                + 1e-6
            || current.distance > params_.corridor_width + 1e-9) {
            continue;
        }
        for (const Eigen::Vector2i& delta : NEIGHBORS) {
            const Eigen::Vector2i next = current.cell + delta;
            if (!grid_cell_traversable(cost_map, next, occupied_threshold)
                || !grid_edge_avoids_corner_cutting(
                    cost_map, current.cell, next, occupied_threshold
                )) {
                continue;
            }
            const double candidate = current.distance
                + delta.cast<double>().norm() * cost_map.geometry.resolution();
            if (candidate > params_.corridor_width + 1e-9) continue;
            GuideFieldCell& next_value = field.cells_[field.index(next)];
            if (candidate + 1e-6
                >= static_cast<double>(next_value.distance_to_reference)) {
                continue;
            }
            next_value.distance_to_reference = static_cast<float>(candidate);
            next_value.nearest_reference_index = current_value.nearest_reference_index;
            open.push({candidate, next});
        }
    }
    field.corridor_cell_count_ = static_cast<size_t>(std::ranges::count_if(
        field.cells_, [&](const GuideFieldCell& cell) {
            return std::isfinite(cell.distance_to_reference)
                && static_cast<double>(cell.distance_to_reference)
                    <= params_.corridor_width + 1e-6;
        }
    ));
    result.success = field.corridor_cell_count_ > 0;
    if (!result.success) result.error = "guide field corridor is empty";
    return result;
}

} // namespace nav_executor
