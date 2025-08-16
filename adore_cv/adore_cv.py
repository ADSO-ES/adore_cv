import rclpy
from rclpy.node import Node

class HelloWorldNode(Node):

    def __init__(self):
        super().__init__('adore_cv')
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info('Hello, World from adore cv!')

def main(args=None):
    rclpy.init(args=args)
    hello_world_node = HelloWorldNode()
    rclpy.spin(hello_world_node)
    hello_world_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

