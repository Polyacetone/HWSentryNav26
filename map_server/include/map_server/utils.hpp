#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace map_server::map_utils {

enum class TerrainType : uint8_t {
    FLAT = 0,
    OBSTACLE = 1,
    SLOPE = 2,
    STEP_L1 = 3,
    STEP_L2 = 4,
    FLY_SLOPE = 5,
    STEP_HIGH = 6
};

struct TerrainMapData {
    int width;
    int height;
    double resolution;
    std::vector<uint8_t> terrain;
    std::vector<uint8_t> direction;
};

struct MapInflationParams {
    double robot_radius_m;
    double cutoff_radius_m;
    double decay_alpha;
    double direction_non_body_magnitude_cap; // (0, 0.9]，须与本体 magnitude=1 保持可分离
    double resolution = 0.0; // map resolution (m/px), must be set before calling inflation functions
};

struct DirectionOverlapPair {
    uint8_t first_label;
    uint8_t second_label;
    size_t cell_count;
};

struct DirectionOverlapSample {
    int x;
    int y;
    uint8_t label_mask;
};

struct DirectionOverlapReport {
    size_t cell_count = 0;
    std::vector<DirectionOverlapPair> pairs;
    std::vector<DirectionOverlapSample> samples;
};

struct NavigationMapData {
    int width;
    int height;
    double resolution;
    cv::Mat cost_map;
    cv::Mat direction_map;
    DirectionOverlapReport direction_overlaps;
};

struct InflatedDirectionField {
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat terrain;
    DirectionOverlapReport overlaps;
};

constexpr bool is_directional_label(uint8_t label) {
    return
        label == static_cast<uint8_t>(TerrainType::SLOPE) ||
        label == static_cast<uint8_t>(TerrainType::STEP_L1) ||
        label == static_cast<uint8_t>(TerrainType::STEP_L2) ||
        label == static_cast<uint8_t>(TerrainType::FLY_SLOPE) ||
        label == static_cast<uint8_t>(TerrainType::STEP_HIGH);
}

TerrainMapData load_terrain_msgpack(const std::string& path);

NavigationMapData load_navigation_maps(
    const std::string& path,
    MapInflationParams inflation_params
);

cv::Mat inflate_cost_map(
    const cv::Mat& source,
    const MapInflationParams& params
);

/// Inflate only connected-component neighborhoods whose support can reach a
/// non-zero output. Falls back to the full-map transform when the aggregate
/// neighborhood area is not smaller than the source map.
cv::Mat inflate_cost_map_bounded(
    const cv::Mat& source,
    const MapInflationParams& params
);

InflatedDirectionField inflate_direction_field(
    const TerrainMapData& data,
    const MapInflationParams& params
);

void build_terrain_3chan(
    const cv::Mat& angle,
    const cv::Mat& magnitude,
    const cv::Mat& terrain,
    cv::Mat& out);

} // namespace map_server::map_utils
