import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input

class HallFinderNode(Node):
    def __init__(self):
        super().__init__('hall_finder_node')

        self.declare_parameter("input", "/percepcion/dbg_image")
        input_topic = self.get_parameter("input").get_parameter_value().string_value
        qos_profile = rclpy.qos.qos_profile_sensor_data

        self.model = load_model('/home/tombae/model/percepcion/model.weights.best.keras')

        self.declare_parameter("class", "class1")
        self.target_class = self.get_parameter("class").get_parameter_value().string_value
        self.class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']

        self.publisher_ = self.create_publisher(Image, 'hall_image', 10)
        self.subscriber = self.create_subscription(Image, input_topic, self.hall_callback, qos_profile)

        # Create a CvBridge object
        self.bridge = CvBridge()

    def preprocess_frame(self, frame):
        input_size = (400, 400)
        image = cv2.resize(frame, input_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

    def add_label_to_frame(self, frame, label):
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def hall_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data)
            image = self.preprocess_frame(frame)

            predictions = self.model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_class_name = self.class_names[predicted_class]

            if predicted_class_name == self.target_class:
                label = f'Detectado: {predicted_class_name}, cerca al destino'
            else:
                label = "Buscando destino"

            self.add_label_to_frame(frame, label)

            # Convert the frame to a ROS Image message and publish
            hall_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(hall_image)

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = HallFinderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
