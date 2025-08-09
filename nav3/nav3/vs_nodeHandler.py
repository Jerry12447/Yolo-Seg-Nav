import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
import os
from ament_index_python.packages import get_package_share_directory

from nav3.controller import PurePursuitController, calculate_target_point
from nav3.yoloseg import YoloSegmentation

class vs_nodeHandler(Node):
    def __init__(self):
        super().__init__('vs_nodeHandler')
        
        # 初始化 CV bridge
        self.bridge = CvBridge()
        
        # 設置 ROI 比例
        self.roi_ratio = 0.35 # 可以調整這個值來改變ROI大小，值越小，關注區域越集中
        
        # 獲取包的共享目錄路徑
        package_share_directory = get_package_share_directory('nav3')
        model_path = os.path.join(package_share_directory, 'models', 'best.pt')
        
        # 初始化 YOLOv8 分割器
        self.yolo_segmenter = YoloSegmentation(model_path=model_path)
        
        # 初始化 Pure Pursuit 控制器
        self.pp_controller = PurePursuitController(k=1.0, lookahead_distance=50)
        
        # 線速度固定為 0.4
        self.linear_velocity_x = 0.4
        
        # 創建訂閱者
        self.rgb_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.rgb_callback,
            10)
        
        self.depth_subscription = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10)
        
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10)
        
        # 創建發布者
        self.cmd_vel_publisher = self.create_publisher(
            JointState,
            '/joint_command',
            10)
        
        self.vis_image_publisher = self.create_publisher(
            Image,
            '/yoloseg',
            10)
        
        # 初始化成員變數
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.target_point = None
        
        # 視覺化窗口
        self.visualization_enabled = True
        
        # 計時器回調
        self.timer = self.create_timer(0.1, self.process_and_control)
        
        self.get_logger().info('Visual Servoing Node 已初始化')
        self.get_logger().info(f'ROI比例設置為: {self.roi_ratio}')
    
    def rgb_callback(self, msg):
        """處理RGB圖像回調
        
        Args:
            msg (Image): ROS圖像訊息
        """
        try:
            # 將ROS圖像轉換為OpenCV格式
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().debug('已接收RGB圖像')
        except Exception as e:
            self.get_logger().error(f'轉換RGB圖像時出錯: {str(e)}')
    
    def depth_callback(self, msg):
        """處理深度圖像回調
        
        Args:
            msg (Image): ROS深度圖像訊息
        """
        try:
            # 將ROS深度圖像轉換為OpenCV格式
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().debug('已接收深度圖像')
        except Exception as e:
            self.get_logger().error(f'轉換深度圖像時出錯: {str(e)}')
    
    def camera_info_callback(self, msg):
        """處理相機資訊回調
        
        Args:
            msg (CameraInfo): ROS相機資訊訊息
        """
        self.camera_info = msg
        self.get_logger().debug('已接收相機資訊')
    
    def process_and_control(self):
        """處理圖像並執行控制邏輯"""
        if self.rgb_image is None:
            self.get_logger().warn('未接收到RGB圖像，跳過處理')
            return
        
        # 執行語意分割，傳入ROI比例
        segmentation_result = self.yolo_segmenter.segment(self.rgb_image, roi_ratio=self.roi_ratio)
        '''
        # 診斷輸出
        self.get_logger().info(f"分割結果鍵: {list(segmentation_result.keys())}")
        if 'Soil' in segmentation_result:
            self.get_logger().info(f"Soil結果鍵: {list(segmentation_result['Soil'].keys())}")
            self.get_logger().info(f"有無土壤輪廓: {len(segmentation_result['Soil']['contours']) > 0}")
        '''
        # 確認是否檢測到土壤
        if segmentation_result['Soil']['contours']:
            # 獲取圖像尺寸
            height, width = self.rgb_image.shape[:2]
            
            # 計算目標點（使用土壤輪廓的加權重心）
            target_x, target_y = calculate_target_point(
                segmentation_result['Soil']['contours'], 
                width, 
                height, 
                self.pp_controller.lookahead_distance
            )
            self.target_point = (target_x, target_y)
            
            # 設定機器人位置（假設在圖像底部中心）
            robot_x = width // 2
            robot_y = height
            
            # 計算角速度
            angular_velocity_z = self.pp_controller.compute(
                robot_x,
                robot_y,
                target_x,
                target_y
            )
            
            # 創建並發布JointState訊息
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.header.frame_id = "base_link"
            
            # 設定關節名稱
            joint_state_msg.name = ['wheel','bearing']
            
            # 設定關節位置
            joint_state_msg.position = [0.0, 0.0]
            
            # 設定關節速度
            joint_state_msg.velocity = [self.linear_velocity_x, angular_velocity_z]
            
            
            self.cmd_vel_publisher.publish(joint_state_msg)
            
            self.get_logger().info(f'線速度={self.linear_velocity_x:.4f}, 角速度={angular_velocity_z:.4f}')
            #self.get_logger().info(f'目標點: ({target_x}, {target_y})')
        else:
            # 如果未檢測到土壤，停止移動
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.header.frame_id = "base_link"
            joint_state_msg.name = ['wheel','bearing']
            joint_state_msg.position = [0.0, 0.0]
            joint_state_msg.velocity = [0.0, 0.0]
            joint_state_msg.effort = [0.0, 0.0]
            
            self.cmd_vel_publisher.publish(joint_state_msg)
            
            self.get_logger().warn('未檢測到土壤，停止移動')
            self.target_point = None
        
        # 可視化結果
        if self.visualization_enabled:
            vis_image = self.yolo_segmenter.visualize(self.rgb_image, segmentation_result)
            
            # 如果有目標點，顯示它
            if self.target_point is not None:
                # 畫大一點的目標點
                cv2.circle(vis_image, self.target_point, 10, (0, 255, 255), -1)
                
                # 從機器人位置（底部中心）到目標點畫一條線
                robot_pos = (vis_image.shape[1] // 2, vis_image.shape[0])
                cv2.line(vis_image, robot_pos, self.target_point, (255, 0, 255), 2)
                
                # 計算並顯示角度
                dx = self.target_point[0] - robot_pos[0]
                dy = robot_pos[1] - self.target_point[1]
                angle = np.arctan2(dx, dy) * 180 / np.pi
                
                cv2.putText(vis_image, 
                           f'target angle: {angle:.2f} rad', 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 255), 
                           2)
                
                # 添加Pure Pursuit參數顯示
                cv2.putText(vis_image, 
                           f'lookahead distance: {self.pp_controller.lookahead_distance}', 
                           (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 255), 
                           2)
            
            # 添加ROI信息顯示
            cv2.putText(vis_image, 
                       f'ROI: {self.roi_ratio}', 
                       (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (255, 255, 0), 
                       2)
            
            # 將視覺化圖像轉換為ROS訊息並發布
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
                vis_msg.header.stamp = self.get_clock().now().to_msg()
                vis_msg.header.frame_id = "camera_link"
                self.vis_image_publisher.publish(vis_msg)
            except Exception as e:
                self.get_logger().error(f'發布視覺化圖像時出錯: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = vs_nodeHandler()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

        
        
        
        
        
