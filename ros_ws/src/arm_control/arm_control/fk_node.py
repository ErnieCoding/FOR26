"""ForwardKinematics node: turns the arm's joint angles into the fingertip's position.

Subscribes:
    arm_state   std_msgs/Float64MultiArray, 6 values: 5 joint angles (deg) and the gripper (%)
Publishes:
    fk_pose     geometry_msgs/PoseStamped: the fingertip's position in the robot's base frame

The simulator draws a pink ball at every position published on fk_pose. If your forward kinematics
is right, the ball sits on the tip of the gripper's fixed finger, wherever the arm goes.
"""
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from arm_control.kinematics import forward_kinematics


class ForwardKinematics(Node):
    def __init__(self):
        super().__init__("forward_kinematics")
        # TODO 1: create a publisher for PoseStamped on "fk_pose" (keep it in self.publisher), and
        # subscribe to "arm_state" (Float64MultiArray) with self.on_arm_state as the callback.
        self.publisher = self.create_publisher(PoseStamped, "fk_pose", 10)

        self.create_subscription(Float64MultiArray, "arm_state", self.on_arm_state, 10)

        self.get_logger().info("ForwardKinematics started.")

    def on_arm_state(self, msg):
        position, direction = forward_kinematics(msg.data)
        # TODO 2: create a PoseStamped, set header.frame_id to "base", copy `position` into
        # pose.position.x, .y and .z (as floats), and publish it.
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.header.frame_id = "base"
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])

        self.publisher.publish(pose)

def main():
    rclpy.init()
    node = ForwardKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
