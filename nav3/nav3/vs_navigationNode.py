import rclpy
from rclpy.node import Node
from nav3.vs_nodeHandler import vs_nodeHandler 

def main(args=None):
    rclpy.init(args=args)
    
    # 初始化vs導航節點
    node = vs_nodeHandler()
    rclpy.logging.get_logger('vs_navigator').info("#[VS] 視覺導航節點初始化 ...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.logging.get_logger('vs_navigator').warn("#[VS] 關閉控制器")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()