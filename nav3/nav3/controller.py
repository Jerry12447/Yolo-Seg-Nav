import numpy as np
import math
import cv2

class PurePursuitController:
    def __init__(self, k=1.0, lookahead_distance=50):
        """Pure Pursuit控制器初始化
        
        Args:
            k (float): 增益係數，控制角速度的強度
            lookahead_distance (float): 前視距離，單位為像素
        """
        self.k = k  # 角速度增益
        self.lookahead_distance = lookahead_distance  # 前視距離（像素）
        
    def compute(self, robot_x, robot_y, target_x, target_y):
        """計算角速度
        
        Args:
            robot_x (float): 機器人當前位置x坐標
            robot_y (float): 機器人當前位置y坐標
            target_x (float): 目標點x坐標
            target_y (float): 目標點y坐標
            
        Returns:
            float: 計算得到的角速度
        """
        # 計算角度差（機器人朝向為y軸負方向，即圖像上方）
        # 計算目標點相對於機器人的位置向量
        dx = target_x - robot_x
        dy = robot_y - target_y  # 注意y軸方向在圖像中是向下的
        
        # 計算目標角度（相對於y軸）
        target_angle = math.atan2(dx, dy)
        
        # 計算目標距離
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 計算曲率 (steering = k * 2 * sin(alpha) / lookahead_distance)
        # 其中alpha是目標角度
        if distance < 1.0:  # 避免距離過小造成不穩定
            return 0.0
        
        # 計算角速度
        angular_velocity = self.k * target_angle
            
        return angular_velocity
        
    def set_lookahead_distance(self, lookahead_distance):
        """設置前視距離
        
        Args:
            lookahead_distance (float): 新的前視距離
        """
        self.lookahead_distance = lookahead_distance
        
    def set_gain(self, k):
        """設置增益係數
        
        Args:
            k (float): 新的增益係數
        """
        self.k = k

def calculate_target_point(contours, image_width, image_height, lookahead_distance=50):
    """計算目標點位置
    
    Args:
        contours (list): 土壤輪廓列表
        image_width (int): 圖像寬度
        image_height (int): 圖像高度
        lookahead_distance (float): 前視距離，單位為像素
        
    Returns:
        tuple: (target_x, target_y) 目標點坐標
    """
    '''
    if not contours:
        # 如果沒有輪廓，返回圖像中心
        return image_width // 2, image_height // 2
    '''
    # 計算所有輪廓的加權重心
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    weighted_x = 0
    weighted_y = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 根據面積大小加權
            weighted_x += cx * (area / total_area)
            weighted_y += cy * (area / total_area)
    
    target_x = int(weighted_x)
    target_y = int(weighted_y)
    
    # 確保目標點是在輪廓內的真實點
    robot_x = image_width // 2
    robot_y = image_height
    
    # 如果計算的目標點與機器人距離太近，則嘗試找一個更遠的點
    distance = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
    if distance < lookahead_distance:
        # 尋找最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 在輪廓上找最遠的點
        max_dist = 0
        farthest_point = (target_x, target_y)
        
        for point in largest_contour[:, 0, :]:
            px, py = point
            dist = math.sqrt((px - robot_x)**2 + (py - robot_y)**2)
            if dist > max_dist and dist <= lookahead_distance * 1.5:
                max_dist = dist
                farthest_point = (px, py)
                
        if max_dist > 0:
            target_x, target_y = farthest_point
    
    return target_x, target_y
