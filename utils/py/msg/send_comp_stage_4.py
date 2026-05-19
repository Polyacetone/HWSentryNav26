import rclpy
from rclpy.node import Node
from interfaces.msg import CompStage

class CompStagePublisher(Node):
    def __init__(self):
        super().__init__('comp_stage_publisher')
        self.comp_stage_publisher_ = self.create_publisher(CompStage, '/serial_bridge/comp_stage', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = CompStage()
        msg.game_progress = 4
        self.comp_stage_publisher_.publish(msg)
        self.get_logger().info('Published CompStage message')

def main(args=None):
    rclpy.init(args=args)
    node = CompStagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()