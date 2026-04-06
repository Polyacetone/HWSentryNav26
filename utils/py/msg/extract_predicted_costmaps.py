import rclpy
from rclpy.node import Node
from interfaces.msg import PredictedCostMaps
from nav_msgs.msg import OccupancyGrid

class PredictedCostmapExtractor(Node):
    def __init__(self):
        super().__init__('predicted_costmap_extractor')
        self.subscription = self.create_subscription(
            PredictedCostMaps,
            '/map_server/predicted_cost_maps',
            self.listener_callback,
            1
        )
        self.publisher = self.create_publisher(OccupancyGrid, '/predicted_costmap', 10)

    def listener_callback(self, msg):
        occupancy_grid = msg.maps[4]
        self.publisher.publish(occupancy_grid)

def main(args=None):
    rclpy.init(args=args)
    extractor = PredictedCostmapExtractor()
    rclpy.spin(extractor)
    extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()