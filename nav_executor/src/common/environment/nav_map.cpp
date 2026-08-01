#include <nav_executor/common/environment/nav_map.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace nav_executor {

namespace {

constexpr uint8_t MAX_NON_BODY_ENCODED_MAGNITUDE = 229;
constexpr double ORIENTATION_TOLERANCE = 1e-9;

void validate_unrotated_orientation(const geometry_msgs::msg::Quaternion& orientation) {
    const double norm = std::hypot(
        std::hypot(orientation.x, orientation.y),
        std::hypot(orientation.z, orientation.w)
    );
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > ORIENTATION_TOLERANCE
        || std::abs(orientation.x / norm) > ORIENTATION_TOLERANCE
        || std::abs(orientation.y / norm) > ORIENTATION_TOLERANCE
        || std::abs(orientation.z / norm) > ORIENTATION_TOLERANCE
        || std::abs(std::abs(orientation.w / norm) - 1.0) > ORIENTATION_TOLERANCE) {
        throw std::invalid_argument(
            "Rotated OccupancyGrid origins are not supported; origin orientation must be a normalized identity quaternion"
        );
    }
}

struct BilinearStencil {
    std::array<Eigen::Vector2i, 4> cells;
    std::array<double, 4> weights;
    std::array<Eigen::Vector2d, 4> weight_gradients;
};

BilinearStencil centered_bilinear_stencil(
    const GridGeometry& geometry, const Eigen::Vector2d& position_map
) {
    Eigen::Vector2d q = (position_map - geometry.origin()) / geometry.resolution()
        - Eigen::Vector2d::Constant(0.5);
    for (int axis = 0; axis < 2; ++axis) {
        const double nearest_integer = std::round(q(axis));
        if (std::abs(q(axis) - nearest_integer) <= 1e-10) q(axis) = nearest_integer;
    }
    const double unclamped_x = q.x();
    const double unclamped_y = q.y();
    q.x() = std::clamp(q.x(), 0.0, static_cast<double>(geometry.width() - 1));
    q.y() = std::clamp(q.y(), 0.0, static_cast<double>(geometry.height() - 1));
    const int x0 = static_cast<int>(std::floor(q.x()));
    const int y0 = static_cast<int>(std::floor(q.y()));
    const int x1 = std::min(x0 + 1, geometry.width() - 1);
    const int y1 = std::min(y0 + 1, geometry.height() - 1);
    const double tx = q.x() - static_cast<double>(x0);
    const double ty = q.y() - static_cast<double>(y0);
    const double inv_resolution = 1.0 / geometry.resolution();
    BilinearStencil stencil {
        .cells = {{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}}},
        .weights = {{
            (1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
            (1.0 - tx) * ty, tx * ty,
        }},
        .weight_gradients = {{
            {-(1.0 - ty) * inv_resolution, -(1.0 - tx) * inv_resolution},
            {(1.0 - ty) * inv_resolution, -tx * inv_resolution},
            {-ty * inv_resolution, (1.0 - tx) * inv_resolution},
            {ty * inv_resolution, tx * inv_resolution},
        }},
    };
    if (unclamped_x < 0.0 || unclamped_x > static_cast<double>(geometry.width() - 1)) {
        for (Eigen::Vector2d& gradient : stencil.weight_gradients) gradient.x() = 0.0;
    }
    if (unclamped_y < 0.0 || unclamped_y > static_cast<double>(geometry.height() - 1)) {
        for (Eigen::Vector2d& gradient : stencil.weight_gradients) gradient.y() = 0.0;
    }
    return stencil;
}

} // anonymous namespace

GridGeometry::GridGeometry(
    const int width, const int height, const double resolution, Eigen::Vector2d origin
): width_(width), height_(height), resolution_(resolution), origin_(std::move(origin)) {
    if (width <= 0 || height <= 0 || !std::isfinite(resolution) || resolution <= 0.0
        || !origin_.allFinite()) {
        throw std::invalid_argument("GridGeometry dimensions, resolution, or origin are invalid");
    }
}

GridGeometry::GridGeometry(const nav_msgs::msg::MapMetaData& metadata):
    GridGeometry(
        static_cast<int>(metadata.width), static_cast<int>(metadata.height),
        static_cast<double>(metadata.resolution),
        {metadata.origin.position.x, metadata.origin.position.y}
    ) {
    validate_unrotated_orientation(metadata.origin.orientation);
}

Eigen::Vector2d GridGeometry::footprint_max() const {
    return origin_ + resolution_ * Eigen::Vector2d(width_, height_);
}

bool GridGeometry::contains_cell(const Eigen::Vector2i& cell) const {
    return cell.x() >= 0 && cell.x() < width_ && cell.y() >= 0 && cell.y() < height_;
}

bool GridGeometry::contains_map_point(const Eigen::Vector2d& point_map) const {
    const Eigen::Vector2d maximum = footprint_max();
    return point_map.allFinite() && point_map.x() >= origin_.x() && point_map.y() >= origin_.y()
        && point_map.x() < maximum.x() && point_map.y() < maximum.y();
}

std::optional<Eigen::Vector2i> GridGeometry::containing_cell(
    const Eigen::Vector2d& point_map
) const {
    if (!contains_map_point(point_map)) return std::nullopt;
    const Eigen::Array2i cell = ((point_map - origin_) / resolution_)
        .array().floor().cast<int>();
    return Eigen::Vector2i(
        std::clamp(cell.x(), 0, width_ - 1),
        std::clamp(cell.y(), 0, height_ - 1)
    );
}

Eigen::Vector2d GridGeometry::cell_center(const Eigen::Vector2i& cell) const {
    if (!contains_cell(cell)) throw std::out_of_range("GridGeometry::cell_center cell is outside map");
    return origin_ + resolution_ * (cell.cast<double>() + Eigen::Vector2d::Constant(0.5));
}

Eigen::Vector2d GridGeometry::clamp_to_footprint(const Eigen::Vector2d& point_map) const {
    if (!point_map.allFinite()) {
        throw std::invalid_argument("GridGeometry::clamp_to_footprint requires a finite point");
    }
    return point_map.cwiseMax(origin_).cwiseMin(footprint_max());
}

Eigen::Vector2d GridGeometry::map_point_to_boundary_grid(
    const Eigen::Vector2d& point_map
) const {
    return (point_map - origin_) / resolution_;
}

bool GridGeometry::same_geometry(const GridGeometry& other) const {
    return width_ == other.width_ && height_ == other.height_
        && resolution_ == other.resolution_ && origin_ == other.origin_;
}

CostMap::CostMap(GridGeometry geometry, std::vector<uint8_t> data):
    geometry(std::move(geometry)), data(std::move(data)) {
    const size_t expected = static_cast<size_t>(this->geometry.width())
        * static_cast<size_t>(this->geometry.height());
    if (this->data.size() != expected) {
        throw std::invalid_argument("CostMap data size does not match geometry");
    }
}

CostMap::CostMap(const nav_msgs::msg::OccupancyGrid& occupancy_grid):
    CostMap(
        GridGeometry(occupancy_grid.info),
        std::vector<uint8_t>(occupancy_grid.data.begin(), occupancy_grid.data.end())
    ) {}

CostMap CostMap::merge(const CostMap& other) const {
    if (!geometry.same_geometry(other.geometry)) {
        throw std::runtime_error("Cannot merge cost maps with different parameters");
    }

    std::vector<uint8_t> merged_data;
    merged_data.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        merged_data.push_back(std::max(data[i], other.data[i]));
    }
    return CostMap(geometry, std::move(merged_data));
}

uint8_t CostMap::raw_cost_at_cell(const Eigen::Vector2i& cell) const {
    if (geometry.contains_cell(cell)) {
        return data[static_cast<size_t>(cell.y()) * static_cast<size_t>(geometry.width())
            + static_cast<size_t>(cell.x())];
    }
    return 255;
}

std::optional<CostMap::CostSample> CostMap::sample_map(
    const Eigen::Vector2d& position_map
) const {
    if (!geometry.contains_map_point(position_map)) return std::nullopt;
    return sample_map_clamped(position_map);
}

CostMap::CostSample CostMap::sample_map_clamped(
    const Eigen::Vector2d& position_map
) const {
    const BilinearStencil stencil = centered_bilinear_stencil(geometry, position_map);
    CostSample sample {.value = 0.0, .gradient = Eigen::Vector2d::Zero()};
    for (size_t i = 0; i < stencil.cells.size(); ++i) {
        const double raw = static_cast<double>(raw_cost_at_cell(stencil.cells[i]));
        sample.value += stencil.weights[i] * raw;
        sample.gradient += stencil.weight_gradients[i] * raw;
    }
    return sample;
}

/*static*/ std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>>
DirectionMap::decode_mat(const cv::Mat& mat) {
    if (mat.type() != CV_8UC3) {
        throw std::runtime_error("Direction map must be of type CV_8UC3");
    }

    std::vector<Eigen::Vector2d> dir;
    std::vector<uint8_t> terrain;
    dir.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));
    terrain.reserve(static_cast<size_t>(mat.cols) * static_cast<size_t>(mat.rows));
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            const cv::Vec3b val = mat.at<cv::Vec3b>(y, x);
            if (val[2] >= TERRAIN_LABEL_COUNT) {
                throw std::runtime_error(
                    "Direction map has invalid terrain label at ("
                    + std::to_string(x) + "," + std::to_string(y) + ")"
                );
            }
            const bool directional_label = val[2] >= static_cast<uint8_t>(TerrainType::SLOPE);
            if (directional_label != (val[1] > 0)) {
                throw std::runtime_error(
                    "Direction map violates label/magnitude invariant at ("
                    + std::to_string(x) + "," + std::to_string(y) + ")"
                );
            }
            if (val[1] > MAX_NON_BODY_ENCODED_MAGNITUDE && val[1] < 255) {
                throw std::runtime_error(
                    "Direction map magnitude is neither capped inflation nor terrain body at ("
                    + std::to_string(x) + "," + std::to_string(y) + ")"
                );
            }
            if (val[1] == 0) {
                dir.emplace_back(0.0, 0.0);
            } else {
                const double angle = static_cast<double>(val[0]) / 255.0 * 2.0 * std::numbers::pi;
                const double mag = static_cast<double>(val[1]) / 255.0;
                dir.emplace_back(std::cos(angle) * mag, std::sin(angle) * mag);
            }
            terrain.push_back(val[2]);
        }
    }
    return {std::move(dir), std::move(terrain)};
}

DirectionMap::DirectionMap(
    const cv::Mat& direction_map, GridGeometry geometry):
    DirectionMap(
        std::move(geometry),
        decode_mat(direction_map)
    ) {
    if (direction_map.cols != this->geometry.width()
        || direction_map.rows != this->geometry.height()) {
        throw std::invalid_argument("Direction map image size does not match geometry");
    }
}

DirectionMap::DirectionMap(
    GridGeometry geometry,
    std::pair<std::vector<Eigen::Vector2d>, std::vector<uint8_t>> decoded):
    DirectionMap(
        std::move(geometry),
        std::move(decoded.first), std::move(decoded.second)
    ) {}

DirectionMap::DirectionMap(
    GridGeometry geometry,
    std::vector<Eigen::Vector2d> dir_data, std::vector<uint8_t> terrain_data):
    geometry(std::move(geometry)),
    data(std::move(dir_data)),
    terrain(terrain_data.empty()
        ? std::vector<uint8_t>(
            static_cast<size_t>(this->geometry.width())
                * static_cast<size_t>(this->geometry.height()),
            static_cast<uint8_t>(TerrainType::FLAT)
        )
        : std::move(terrain_data)) {
    const size_t expected = static_cast<size_t>(this->geometry.width())
        * static_cast<size_t>(this->geometry.height());
    if (this->data.size() != expected) {
        throw std::runtime_error("DirectionMap data size does not match width*height");
    }
    if (this->terrain.size() != expected) {
        throw std::runtime_error("DirectionMap terrain size does not match width*height");
    }
    for (size_t i = 0; i < this->data.size(); ++i) {
        if (!this->data[i].allFinite() || this->terrain[i] >= TERRAIN_LABEL_COUNT) {
            throw std::runtime_error("DirectionMap contains invalid cell data");
        }
    }
}

Eigen::Vector2d DirectionMap::raw_direction_at_cell(const Eigen::Vector2i& cell) const {
    if (geometry.contains_cell(cell)) {
        return data[static_cast<size_t>(cell.y()) * static_cast<size_t>(geometry.width())
            + static_cast<size_t>(cell.x())];
    }
    return {0, 0};
}

double DirectionMap::raw_magnitude_at_cell(const Eigen::Vector2i& cell) const {
    return raw_direction_at_cell(cell).norm();
}

bool DirectionMap::is_terrain_body_cell(const Eigen::Vector2i& cell) const {
    return raw_magnitude_at_cell(cell) > TERRAIN_BODY_MAGNITUDE_THRESHOLD;
}

uint8_t DirectionMap::terrain_label_at_cell(const Eigen::Vector2i& cell) const {
    if (geometry.contains_cell(cell)) {
        return terrain[static_cast<size_t>(cell.y()) * static_cast<size_t>(geometry.width())
            + static_cast<size_t>(cell.x())];
    }
    return static_cast<uint8_t>(TerrainType::OBSTACLE);
}

std::optional<DirectionMap::DirectionSample> DirectionMap::sample_map(
    const Eigen::Vector2d& position_map
) const {
    if (!geometry.contains_map_point(position_map)) return std::nullopt;
    return sample_map_clamped(position_map);
}

DirectionMap::DirectionSample DirectionMap::sample_map_clamped(
    const Eigen::Vector2d& position_map
) const {
    const BilinearStencil stencil = centered_bilinear_stencil(geometry, position_map);
    DirectionSample result {
        .value = Eigen::Vector2d::Zero(),
        .jacobian = Eigen::Matrix2d::Zero(),
    };
    for (size_t i = 0; i < stencil.cells.size(); ++i) {
        const Eigen::Vector2d raw = raw_direction_at_cell(stencil.cells[i]);
        result.value += stencil.weights[i] * raw;
        result.jacobian += raw * stencil.weight_gradients[i].transpose();
    }
    return result;
}

DirectionMap::LabelWeights DirectionMap::label_weights_clamped(
    const Eigen::Vector2d& position_map
) const {
    const BilinearStencil stencil = centered_bilinear_stencil(geometry, position_map);
    LabelWeights result;
    for (size_t i = 0; i < stencil.cells.size(); ++i) {
        const size_t label = terrain_label_at_cell(stencil.cells[i]);
        result.weights[label] += stencil.weights[i];
        result.dweights[label] += stencil.weight_gradients[i];
    }
    return result;
}

const TraversalMode* TerrainTraversalConstraints::selected_mode(const uint8_t label, const bool is_up) const {
    if (label < static_cast<uint8_t>(TerrainType::SLOPE) || label >= TERRAIN_LABEL_COUNT) return nullptr;
    const auto& mode = is_up ? selected_modes[label - 2].up : selected_modes[label - 2].down;
    return mode ? &*mode : nullptr;
}

TerrainTraversalConstraints build_terrain_traversal_constraints(
    const DirectionMap& direction_map,
    const TraversalConfiguration& configuration,
    const PerformanceState performance
) {
    TerrainTraversalConstraints constraints;
    std::vector<uint8_t> blocked(direction_map.data.size(), 0);
    for (uint8_t label = static_cast<uint8_t>(TerrainType::SLOPE); label < TERRAIN_LABEL_COUNT; ++label) {
        const auto select = [&](const std::vector<TraversalMode>& modes)
            -> std::optional<TraversalMode> {
            for (const TraversalMode& mode : modes) {
                if (!mode.requires_high_performance || performance.high_performance) {
                    return mode;
                }
            }
            return std::nullopt;
        };
        auto& selected = constraints.selected_modes[label - 2];
        selected.up = select(configuration.directional_labels[label - 2].up);
        selected.down = select(configuration.directional_labels[label - 2].down);
        constraints.rules[label] = {
            .forward_allowed = selected.up.has_value(),
            .backward_allowed = selected.down.has_value(),
        };
    }
    for (size_t i = 0; i < blocked.size(); ++i) {
        const uint8_t label = direction_map.terrain[i];
        const TerrainRule& rule = constraints.rules[label];
        if (label >= static_cast<uint8_t>(TerrainType::SLOPE)
            && !rule.forward_allowed && !rule.backward_allowed) {
            blocked[i] = static_cast<uint8_t>(std::clamp(
                std::lround(255.0 * direction_map.data[i].norm()), 0L, 255L
            ));
        }
    }
    constraints.blocked_cost_layer = std::make_shared<CostMap>(
        direction_map.geometry, std::move(blocked)
    );
    return constraints;
}

} // namespace nav_executor
