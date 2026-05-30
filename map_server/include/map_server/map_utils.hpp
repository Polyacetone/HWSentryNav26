#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace map_server::map_utils {

struct TerrainMapData {
    int width;
    int height;
    double resolution;
    std::vector<uint8_t> terrain;
    std::vector<uint8_t> direction;
};

struct MapInflationParams {
    int robot_radius_px;
    int cutoff_radius_px;
    double decay_alpha;
};

constexpr uint8_t TERRAIN_FLAT = 0;
constexpr uint8_t TERRAIN_OBSTACLE = 1;
constexpr uint8_t TERRAIN_SLOPE = 2;
constexpr uint8_t TERRAIN_STEP_L1 = 3;
constexpr uint8_t TERRAIN_STEP_L2 = 4;
constexpr uint8_t TERRAIN_FLY_SLOPE = 5;
constexpr uint8_t TERRAIN_STEP_HIGH = 6;

constexpr bool is_directional_label(uint8_t label) {
    return label >= TERRAIN_SLOPE && label <= TERRAIN_STEP_HIGH;
}

TerrainMapData load_terrain_msgpack(const std::string& path);

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
