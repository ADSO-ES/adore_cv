import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import pandas as pd
import rclpy
from rclpy.node import Node


class YoloCnnDetector(Node):
    def __init__(self, yolo_path, cnn_path, input_size=(64, 64), cam_index=0):
        super().__init__("yolo_cnn_detector")

        # Load models once
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_path)
        self.get_logger().info("Loading CNN model...")
        self.cnn_model = tf.keras.models.load_model(cnn_path)

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera")
            rclpy.shutdown()
            return

        self.input_size = input_size

        # Create timer to run detection at ~10Hz
        self.timer = self.create_timer(0.1, self.detect_callback)

    def detect_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        results = self.yolo_model(frame)
        combined_results = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.yolo_model.names[class_id]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, self.input_size)
            input_tensor = np.expand_dims(resized, axis=0)

            cnn_pred = self.cnn_model.predict(input_tensor, verbose=0)[0]
            cnn_class = int(np.argmax(cnn_pred))
            cnn_conf = float(np.max(cnn_pred))

            warning = int(class_id != cnn_class)

            combined_results.append({
                "yolo_class_name": class_name,
                "yolo_conf": round(conf, 3),
                "cnn_class": cnn_class,
                "cnn_conf": round(cnn_conf, 3),
                "warning": warning
            })

        if combined_results:
            df = pd.DataFrame(combined_results)
            # Publish to rosout
            self.get_logger().info(f"Detections:\n{df.to_string(index=False)}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloCnnDetector(
        yolo_path="/ros2_ws/src/adore_cv/adore_cv/yolo_sign_obstacles.pt",
        cnn_path="/ros2_ws/src/adore_cv/adore_cv/CNN_sign_obstacles.h5",
        cam_index=2
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
