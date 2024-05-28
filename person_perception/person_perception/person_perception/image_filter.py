import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy

class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_filter')
    self.declare_parameter("input_image", "image_raw")
    input_topic = self.get_parameter(
        "input_image").get_parameter_value().string_value
    qos_profile = rclpy.qos.qos_profile_sensor_data

    self._pub = self.create_publisher(Image, "image_pr", 10)
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(Image, input_topic, self.listener_callback, qos_profile=qos_profile)
    self.subscription # prevent unused variable warning

    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    self.frame_received = False

  def listener_callback(self, data):
    """
    Callback function.
    """
    if not self.frame_received:
        self.get_logger().info('Receiving video frame')
        self.frame_received = True
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    # Apply saturation improvement
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(1.5 * hsv[:,:,1], 0, 255)
    saturated_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Convert the processed image back to ROS Image message
    image_pr = self.br.cv2_to_imgmsg(saturated_image)

    # Publish the processed image
    self._pub.publish(image_pr)

def main(args=None):

  # Initialize the rclpy library
  rclpy.init(args=args)

  # Create the node
  image_filter = ImageSubscriber()

  # Spin the node so the callback function is called.
  rclpy.spin(image_filter)

  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_filter.destroy_node()

  # Shutdown the ROS client library for Python
  rclpy.shutdown()

if __name__ == '__main__':
  main()
