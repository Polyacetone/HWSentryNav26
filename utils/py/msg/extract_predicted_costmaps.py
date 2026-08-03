import rclpy
from rclpy.node import Node
from interfaces.msg import CostMaps
from nav_msgs.msg import OccupancyGrid

class PredictedCostmapExtractor(Node):
    def __init__(self):
        super().__init__('predicted_costmap_extractor')
        self.subscription = self.create_subscription(
            CostMaps,
            '/map_server/local_cost_maps',
            self.listener_callback,
            1
        )
        self.publisher0 = self.create_publisher(OccupancyGrid, '/predicted_costmap_0', 10)
        self.publisher19 = self.create_publisher(OccupancyGrid, '/predicted_costmap_19', 10)

    def listener_callback(self, msg):
        self.publisher0.publish(msg.maps[0])
        self.publisher19.publish(msg.maps[19])

def main(args=None):
    rclpy.init(args=args)
    extractor = PredictedCostmapExtractor()
    rclpy.spin(extractor)
    extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()