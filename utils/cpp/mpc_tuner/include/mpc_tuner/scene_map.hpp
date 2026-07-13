#pragma once

#include <filesystem>

#include <map_server/utils.hpp>

#include <mpc_tuner/types.hpp>

namespace mpc_tuner {

struct LoadedSceneMap {
    MapSnapshot snapshot;
    nav_executor::CostMap::ConstPtr cost_map;
    nav_executor::DirectionMap::ConstPtr direction_map;
};

[[nodiscard]] LoadedSceneMap load_scene_map(
    const std::filesystem::path& terrain_map_path,
    map_server::map_utils::MapInflationParams inflation
);

} // namespace mpc_tuner
