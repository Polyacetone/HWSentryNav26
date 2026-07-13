#include <mpc_tuner/scene_map.hpp>

#include <opencv2/core.hpp>

namespace mpc_tuner {

LoadedSceneMap load_scene_map(
    const std::filesystem::path& terrain_map_path,
    map_server::map_utils::MapInflationParams inflation
) {
    auto maps = map_server::map_utils::load_navigation_maps(terrain_map_path.string(), inflation);

    MapSnapshot snapshot {
        .width = maps.width,
        .height = maps.height,
        .resolution = maps.resolution,
        .origin_x = 0.0,
        .origin_y = 0.0,
        .global_cost_data = {},
        .direction_image_data = {},
    };
    snapshot.global_cost_data.assign(maps.cost_map.datastart, maps.cost_map.dataend);
    snapshot.direction_image_data.assign(maps.direction_map.datastart, maps.direction_map.dataend);

    return {
        .snapshot = snapshot,
        .cost_map = std::make_shared<const nav_executor::CostMap>(
            snapshot.width, snapshot.height, snapshot.resolution,
            snapshot.origin_x, snapshot.origin_y, snapshot.global_cost_data
        ),
        .direction_map = std::make_shared<const nav_executor::DirectionMap>(
            maps.direction_map, snapshot.resolution, snapshot.origin_x, snapshot.origin_y
        ),
    };
}

} // namespace mpc_tuner
