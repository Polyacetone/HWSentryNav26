#pragma once

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
    double resolution = 0.0; // map resolution (m/px), must be set before calling inflation functions
};

struct NavigationMapData {
    int width;
    int height;
    double resolution;
    cv::Mat cost_map;
    cv::Mat direction_map;
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

void inflate_direction_field(
    const TerrainMapData& data,
    const MapInflationParams& params,
    cv::Mat& out_angle,
    cv::Mat& out_magnitude);

void build_terrain_3chan(
    const cv::Mat& angle,
    const cv::Mat& magnitude,
    const cv::Mat& terrain,
    cv::Mat& out);

} // namespace map_server::map_utils
