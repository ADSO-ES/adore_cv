import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import pandas as pd
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from std_msgs.msg import Header
# from geometry_msgs.msg import Pose2D
class YoloCnnDetector(Node):
    def __init__(self, yolo_path, cnn_path, input_size=(64, 64), cam_index=0):
        super().__init__("yolo_cnn_detector")
        self.publisher_ = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
        timer_period = 0.5  # 2Hz publish rate
        self.timer = self.create_timer(timer_period, self.detect_callback)
        # Load models once
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_path)
        self.get_logger().info("Loading CNN model...")
        self.cnn_model = tf.keras.models.load_model(cnn_path)

        # Open camera
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera")
            rclpy.shutdown()
            return

        self.input_size = input_size

        # Create timer to run detection at ~10Hz
        #self.timer = self.create_timer(0.1, self.detect_callback)

    def detect_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        results = self.yolo_model(frame)
        combined_results = []

        detection_array = Detection2DArray()
        detection_array.header = Header()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_frame"

        detections = []
        #detection_array.header.seq = 43
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # CNN classification
            resized = cv2.resize(crop, self.input_size)
            input_tensor = np.expand_dims(resized, axis=0)

            cnn_pred = self.cnn_model.predict(input_tensor, verbose=0)[0]
            cnn_class = int(np.argmax(cnn_pred))
            cnn_conf = float(np.max(cnn_pred))

            # Generate Warning
            warning = int(class_id != cnn_class)

            # Building Detection2D
            detection = Detection2D()
            detection.header = detection_array.header
            
            yolo_hyp = ObjectHypothesisWithPose()
            yolo_hyp.hypothesis.class_id = str(class_id)
            print(f"class id:{conf}")
            yolo_hyp.hypothesis.score = conf
            detection.results.append(yolo_hyp)

            # Bounding box
            bbox = BoundingBox2D()
            # bbox.center = Pose2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            #bbox.center.theta = 0.0  # orientation not used
            bbox.size_x = float((x2 - x1))
            bbox.size_y = float((y2 - y1))
            detection.bbox = bbox
            
            detections.append(detection)

            combined_results.append({
                "yolo_class_name": class_name,
                "yolo_conf": round(conf, 3),
                "cnn_class": cnn_class,
                "cnn_conf": round(cnn_conf, 3),
                "warning": warning
            })
        detection_array.detections = detections
        if combined_results:
            df = pd.DataFrame(combined_results)
            # Publish to rosout
            # self.get_logger().info(f"Detections:\n{df.to_string(index=False)}")
        # Publish only if detections exist
        if len(detection_array.detections) > 0:
            self.publisher_.publish(detection_array)
            #self.get_logger().info(f"Published {len(detection_array.detections)} detections")
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloCnnDetector(
        yolo_path="/ros2_ws/src/adore_cv/adore_cv/yolo_sign_obstacles.pt",
        cnn_path="/ros2_ws/src/adore_cv/adore_cv/CNN_sign_obstacles.h5",
        cam_index=1
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
