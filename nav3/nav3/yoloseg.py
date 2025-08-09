import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from nav3.geometric import lineIntersectImgSides, lineIntersectImgUpDown, getImgLineUpDown

class YoloSegmentation:
    def __init__(self, model_path='./models/best.pt', device=None):
        """初始化YOLOv8語意分割模型
        
        Args:
            model_path (str): 模型路徑
            device (str, optional): 運行設備，如 'cuda:0' 或 'cpu'
        """
        # 確保模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 加載模型
        self.model = YOLO(model_path)
        
        # 設置運行設備
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 類別映射 (根據您的模型調整)
        self.class_names = {
            0: 'Soil',
            1: 'ditch'
        }
        
    def segment(self, image, roi_ratio):
        """執行語意分割
        
        Args:
            image (numpy.ndarray): 輸入RGB圖像
            roi_ratio (float): ROI區域佔圖像的比例，介於0-1之間
            
        Returns:
            dict: 包含分割結果的字典，包括掩碼和邊界
        """
        # 獲取圖像尺寸
        height, width = image.shape[:2]
        
        # 計算梯形ROI區域參數（根據土壤分佈特徵調整）
        # 下底（近處）寬度 - 較寬以覆蓋更多近處區域
        bottom_width = int(width * roi_ratio * 1.5)  
        # 上底（遠處）寬度 - 較窄以聚焦於遠處中心
        top_width = int(width * roi_ratio * 0.7)
        # 梯形高度 - 從底部到中間偏上位置
        roi_height = int(height * 0.8)
        # 頂部位置調整 - 使梯形頂部更靠近地平線
        top_y_offset = int(height * 0.2)
        
        # 計算四個頂點坐標
        bottom_left = (max(0, width // 2 - bottom_width // 2), height)  # 左下
        bottom_right = (min(width, width // 2 + bottom_width // 2), height)  # 右下
        top_left = (max(0, width // 2 - top_width // 2), top_y_offset)  # 左上
        top_right = (min(width, width // 2 + top_width // 2), top_y_offset)  # 右上
        
        # 設置梯形ROI掩碼
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_points = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], 255)
        
        # 存儲ROI頂點，用於可視化
        roi_vertices = {
            'bottom_left': bottom_left,
            'bottom_right': bottom_right,
            'top_left': top_left,
            'top_right': top_right,
            'shape': 'trapezoid'
        }
        
        # 執行推理
        results = self.model.predict(image, device=self.device, verbose=False)
        
        # 初始化結果字典
        segmentation_result = {
            'Soil': {
                'mask': None,
                'contours': [],
                'center_line': None,
                'center_x': None,
                'center_points': []  # 存儲多個高度的中心點
            },
            'ditch': {
                'mask': None,
                'contours': []
            },
            'roi': roi_vertices
        }
        
        # 處理結果
        if len(results) > 0:
            result = results[0]
            
            # 獲取掩碼
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data
                
                # 獲取類別
                cls = result.boxes.cls.cpu().numpy().astype(int)
                
                # 處理每個類別
                for c in [0, 1]:  # Soil和ditch
                    class_name = self.class_names[c]
                    
                    # 初始化該類別的掩碼
                    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    
                    # 合併同一類別的所有掩碼
                    for i, mask in enumerate(masks):
                        if i < len(cls) and cls[i] == c:
                            # 將掩碼轉換為numpy數組並調整大小
                            mask_np = mask.cpu().numpy()
                            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                            
                            # 二值化
                            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                            
                            # 合併掩碼
                            combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                    
                    # 儲存掩碼 (對於Soil類別，應用ROI掩碼)
                    if class_name == 'Soil':
                        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)
                    
                    segmentation_result[class_name]['mask'] = combined_mask
                    
                    # 如果有掩碼，計算輪廓
                    if np.any(combined_mask > 0):
                        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # 過濾掉太小的輪廓
                            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
                            segmentation_result[class_name]['contours'] = contours
        
        # 計算土壤中心點 (使用輪廓加權重心)
        if segmentation_result['Soil']['contours']:
            # 使用輪廓加權重心計算中心點
            contours = segmentation_result['Soil']['contours']
            
            # 在不同高度上計算中心點
            center_points = []
            # 從底部到頂部以固定間隔選取多個高度
            height_steps = 8  # 選取的高度數量
            for i in range(height_steps):
                # 計算當前高度
                current_height = int(height * (0.9 - i * 0.1))
                if current_height < top_y_offset:
                    break
                
                # 根據當前高度過濾輪廓點
                filtered_points = []
                y_range = 20  # 高度範圍
                
                for cnt in contours:
                    # 提取輪廓中在當前高度附近的點
                    points_at_height = []
                    for point in cnt[:, 0, :]:
                        if current_height - y_range <= point[1] <= current_height + y_range:
                            points_at_height.append(point[0])  # 只保留 x 座標
                    
                    if points_at_height:
                        avg_x = sum(points_at_height) / len(points_at_height)
                        filtered_points.append((avg_x, current_height))
                
                if filtered_points:
                    # 計算當前高度的加權平均中心點
                    avg_x = sum(p[0] for p in filtered_points) / len(filtered_points)
                    center_points.append((int(avg_x), current_height))
            
            # 存儲中心點列表
            segmentation_result['Soil']['center_points'] = center_points
            
            # 如果有足夠的中心點，則計算中心線
            if len(center_points) >= 2:
                # 提取中心點的 x 和 y 座標
                x_coords = [p[0] for p in center_points]
                y_coords = [p[1] for p in center_points]
                
                # 使用線性回歸計算中心線方程
                try:
                    coefficients = np.polyfit(y_coords, x_coords, 1)
                    m, b = coefficients
                    segmentation_result['Soil']['center_line'] = (m, b)
                    print(f"通過加權中心點計算得到中心線: m={m}, b={b}")
                except Exception as e:
                    print(f"計算中心線時出錯: {str(e)}")
                    segmentation_result['Soil']['center_line'] = None
            
            # 設置主要中心點 (在圖像較低位置)
            center_y = int(height * 0.7)  # 偏下位置
            
            # 如果已計算出中心線，則使用中心線方程計算中心點
            if segmentation_result['Soil']['center_line'] is not None:
                m, b = segmentation_result['Soil']['center_line']
                center_x = self.get_x_from_line(m, b, center_y)
                segmentation_result['Soil']['center_x'] = center_x
                print(f"使用中心線計算得到中心點: {center_x} (y={center_y})")
            # 否則使用輪廓加權重心
            elif center_points:
                # 找到最接近指定高度的中心點
                closest_point = min(center_points, key=lambda p: abs(p[1] - center_y))
                center_x = closest_point[0]
                segmentation_result['Soil']['center_x'] = center_x
                print(f"使用最近的中心點: {center_x} (y={closest_point[1]})")
            # 退化方案：直接計算所有輪廓的加權重心
            else:
                total_area = sum(cv2.contourArea(cnt) for cnt in contours)
                weighted_x = 0
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        # 根據面積大小加權
                        weighted_x += cx * (area / total_area)
                
                center_x = int(weighted_x)
                segmentation_result['Soil']['center_x'] = center_x
                print(f"使用輪廓加權重心計算得到中心點: {center_x}")
        
        return segmentation_result
    
    def calculate_centerline(self, mask, contours, width, height):
        """計算土壤區域的中心線
        
        Args:
            mask (numpy.ndarray): 土壤掩碼
            contours (list): 輪廓列表
            width (int): 圖像寬度
            height (int): 圖像高度
            
        Returns:
            tuple: (斜率, 截距) 表示中心線，如果無法計算則返回None
        """
        if not contours:
            return None
            
        # 選擇最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 如果面積太小，則跳過
        if cv2.contourArea(largest_contour) < 500:
            return None
            
        # 計算骨架/中心線
        # 方法1: 使用細化算法
        skeleton = self.get_skeleton(mask)
        
        # 從骨架提取線段
        lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            # 方法2: 通過在不同高度取樣輪廓中心
            return self.get_centerline_by_sampling(largest_contour, height, width)
        
        # 合併線段
        merged_lines = self.merge_lines(lines, height, width)
        
        if merged_lines:
            # 取第一條合併後的線
            return merged_lines[0]
        
        return None
    
    def get_skeleton(self, mask):
        """獲取掩碼的骨架
        
        Args:
            mask (numpy.ndarray): 二值掩碼
            
        Returns:
            numpy.ndarray: 骨架圖像
        """
        # 確保掩碼是二值的
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 使用形態學操作獲取骨架 (替代 ximgproc.thinning)
        # 創建結構元素
        kernel = np.ones((3, 3), np.uint8)
        
        # 骨架提取
        skeleton = np.zeros_like(binary)
        eroded = np.zeros_like(binary)
        temp = np.zeros_like(binary)
        
        # 複製二值圖像
        img = binary.copy()
        
        # 迭代直到圖像完全侵蝕
        while cv2.countNonZero(img) > 0:
            # 侵蝕操作
            cv2.erode(img, kernel, eroded)
            
            # 開運算
            cv2.dilate(eroded, kernel, temp)
            cv2.subtract(img, temp, temp)
            
            # 添加到骨架
            cv2.bitwise_or(skeleton, temp, skeleton)
            
            # 更新圖像
            img = eroded.copy()
        
        return skeleton
    
    def merge_lines(self, lines, height, width):
        """合併檢測到的線段
        
        Args:
            lines (numpy.ndarray): 檢測到的線段
            height (int): 圖像高度
            width (int): 圖像寬度
            
        Returns:
            list: 合併後的線 [(m, b), ...]，其中m為斜率，b為截距
        """
        if lines is None:
            print("沒有檢測到線段")
            return []
            
        # 獲取所有線的斜率和截距
        mb_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 避免垂直線
            if x2 - x1 == 0:
                print(f"跳過垂直線: ({x1},{y1}) - ({x2},{y2})")
                continue
              
            # 計算斜率和截距
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # 添加一些容錯（斜率不能太大）
            if abs(m) > 100:
                print(f"斜率太大: {m}")
                continue
                
            # 檢查線是否在合理範圍內
            try:
                left_x, right_x = lineIntersectImgSides(m, b, width)
                top_y, bottom_y = lineIntersectImgUpDown(m, b, height)
                
                # 如果線段在圖像範圍內，則加入
                if (0 <= left_x <= width and 0 <= right_x <= width and 
                    0 <= top_y <= width and 0 <= bottom_y <= width):
                    mb_lines.append((m, b))
                    print(f"添加線段: 斜率={m}, 截距={b}")
                else:
                    print(f"線段超出範圍: left_x={left_x}, right_x={right_x}, top_y={top_y}, bottom_y={bottom_y}")
            except Exception as e:
                print(f"線段處理出錯: {e}")
        
        # 如果沒有合適的線，返回空列表
        if not mb_lines:
            print("沒有合適的線段")
            return []
        
        # 簡單方法：返回所有線的平均斜率和截距
        avg_m = sum(m for m, _ in mb_lines) / len(mb_lines)
        avg_b = sum(b for _, b in mb_lines) / len(mb_lines)
        
        print(f"最終中心線: 斜率={avg_m}, 截距={avg_b}")
        return [(avg_m, avg_b)]
    
    def get_centerline_by_sampling(self, contour, height, width):
        """通過在不同高度採樣輪廓中心來計算中心線
        
        Args:
            contour (numpy.ndarray): 輪廓點
            height (int): 圖像高度
            width (int): 圖像寬度
            
        Returns:
            tuple: (斜率, 截距) 表示中心線，如果無法計算則返回None
        """
        # 在不同高度採樣點
        sample_points = []
        step = height // 10  # 採樣10個點
        
        for y in range(step, height - step, step):
            # 找到在當前高度的所有輪廓點
            points_at_height = [pt[0][0] for pt in contour if pt[0][1] == y]
            
            if points_at_height:
                # 取平均值作為中心
                center_x = sum(points_at_height) / len(points_at_height)
                sample_points.append((center_x, y))
        
        # 如果樣本點太少，無法確定線
        if len(sample_points) < 2:
            return None
        
        # 使用線性回歸計算中心線
        x_coords, y_coords = zip(*sample_points)
        coefficients = np.polyfit(x_coords, y_coords, 1)
        m, b = coefficients
        
        # 斜率變換 (y = mx + b 到 x = my + b)
        if m != 0:
            m_new = 1 / m
            b_new = -b / m
            return (m_new, b_new)
        else:
            # 垂直線，返回特殊值
            return (1000, np.mean(x_coords))  # 非常大的斜率
    
    def get_x_from_line(self, m, b, y):
        """給定y坐標，計算線上的x坐標
        
        Args:
            m (float): 斜率
            b (float): 截距
            y (float): y坐標
            
        Returns:
            float: x坐標
        """
        return m * y + b
    
    def visualize(self, image, segmentation_result):
        """可視化分割結果
        
        Args:
            image (numpy.ndarray): 原始圖像
            segmentation_result (dict): 分割結果
            
        Returns:
            numpy.ndarray: 可視化圖像
        """
        # 複製原始圖像以避免修改
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        # 繪製土壤掩碼
        if segmentation_result['Soil']['mask'] is not None:
            Soil_mask = segmentation_result['Soil']['mask']
            Soil_overlay = np.zeros_like(vis_image)
            Soil_overlay[Soil_mask > 0] = [0, 255, 0]  # 綠色
            vis_image = cv2.addWeighted(vis_image, 1, Soil_overlay, 0.3, 0)
            
            # 繪製土壤輪廓
            if segmentation_result['Soil']['contours']:
                cv2.drawContours(vis_image, segmentation_result['Soil']['contours'], -1, (0, 255, 0), 2)
                
        # 繪製溝渠掩碼
        if segmentation_result['ditch']['mask'] is not None:
            ditch_mask = segmentation_result['ditch']['mask']
            ditch_overlay = np.zeros_like(vis_image)
            ditch_overlay[ditch_mask > 0] = [0, 0, 255]  # 紅色
            vis_image = cv2.addWeighted(vis_image, 1, ditch_overlay, 0.3, 0)
            
            # 繪製溝渠輪廓
            if segmentation_result['ditch']['contours']:
                cv2.drawContours(vis_image, segmentation_result['ditch']['contours'], -1, (0, 0, 255), 2)
        
        # 繪製ROI區域 (在土壤和溝渠之後繪製，以便可以看到邊界)
        if 'roi' in segmentation_result:
            roi = segmentation_result['roi']
            if roi['shape'] == 'trapezoid':
                # 繪製梯形ROI
                cv2.line(vis_image, roi['bottom_left'], roi['bottom_right'], (255, 255, 0), 2)
                cv2.line(vis_image, roi['bottom_right'], roi['top_right'], (255, 255, 0), 2)
                cv2.line(vis_image, roi['top_right'], roi['top_left'], (255, 255, 0), 2)
                cv2.line(vis_image, roi['top_left'], roi['bottom_left'], (255, 255, 0), 2)
                
                # 添加ROI角點標記
                cv2.circle(vis_image, roi['bottom_left'], 5, (255, 255, 0), -1)
                cv2.circle(vis_image, roi['bottom_right'], 5, (255, 255, 0), -1)
                cv2.circle(vis_image, roi['top_left'], 5, (255, 255, 0), -1)
                cv2.circle(vis_image, roi['top_right'], 5, (255, 255, 0), -1)
        
        # 繪製多個高度的中心點
        if 'center_points' in segmentation_result['Soil'] and segmentation_result['Soil']['center_points']:
            center_points = segmentation_result['Soil']['center_points']
            
            # 繪製中心點
            for i, (x, y) in enumerate(center_points):
                # 使用不同的顏色以區分不同高度的點
                color = (0, 165 + (90 * i // len(center_points)), 255)
                cv2.circle(vis_image, (int(x), int(y)), 4, color, -1)
        
        # 標記機器人位置（底部中心）
        robot_x = width // 2
        robot_y = height
        cv2.circle(vis_image, (robot_x, robot_y), 8, (0, 0, 255), -1)
        
        return vis_image
