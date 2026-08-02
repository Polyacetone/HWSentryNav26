#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nav_executor/path_planner/search/reference_path.hpp>

namespace nav_executor {

struct GuideFieldCell {
    float distance_to_reference = std::numeric_limits<float>::infinity();
    uint32_t nearest_reference_index = std::numeric_limits<uint32_t>::max();
};

class GuideField {
public:
    [[nodiscard]] bool ready() const { return geometry_.has_value(); }
    [[nodiscard]] bool contains(const Eigen::Vector2i& cell) const;
    [[nodiscard]] bool contains_map(const Eigen::Vector2d& point_map) const;
    [[nodiscard]] const GuideFieldCell* at_cell(const Eigen::Vector2i& cell) const;
    [[nodiscard]] const GuideFieldCell* at_map(const Eigen::Vector2d& point_map) const;
    [[nodiscard]] size_t corridor_cell_count() const { return corridor_cell_count_; }

private:
    friend class GuideFieldBuilder;
    [[nodiscard]] size_t index(const Eigen::Vector2i& cell) const;

    std::optional<GridGeometry> geometry_;
    std::vector<GuideFieldCell> cells_;
    size_t corridor_cell_count_ = 0;
};

class GuideFieldBuilder {
public:
    struct Params {
        double corridor_width = 2.0;
        double start_bulb_radius = 2.3;
    };

    struct Result {
        GuideField field;
        bool success = false;
        std::string error;
    };

    explicit GuideFieldBuilder(Params params) : params_(params) {}

    [[nodiscard]] Result build(
        const ReferencePath& reference_path,
        const CostMap& cost_map,
        int occupied_threshold
    ) const;

private:
    Params params_;
};

} // namespace nav_executor
