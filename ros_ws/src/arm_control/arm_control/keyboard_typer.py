"""KeyboardTyper node: presses keys when it is told a letter or a word.

Subscribes:
    key           std_msgs/String: one letter, for example "p", or a word, for example "pool". Every
                  letter must be on the selected keyboard. Spaces are ignored, capitals count as lower case.
    arm_state     std_msgs/Float64MultiArray: where the arm is now
Publishes:
    arm_commands  std_msgs/Float64MultiArray, 6 values: 5 joint angles (deg) and the gripper (%)
Parameters:
    keyboard      "sim" (default): the simulated keys a, s, d. "real": our keyboard's p, o, l.
    move_time     seconds for each move (default MOVE_TIME). Use a larger value on the real robot at first.

For each letter the fingertip moves to a point above the key, down onto it, back up, and then home.
A word is typed letter by letter. A message that arrives while a letter or word is being typed is ignored.
At each of these points the arm is held briefly while the measured joint error is fed back into the
command, so the arm does not stop short under gravity or the key's spring.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from arm_control.kinematics import inverse_kinematics

# Where the top of each key is, in the robot's base frame (metres).
SIM_KEY_POSITIONS = {    # the simulated keyboard in the "keyboard" scene
    "a": [0.300, 0.070, -0.012],
    "s": [0.300, 0.000, -0.012],
    "d": [0.300, -0.070, -0.012],
}
REAL_KEY_POSITIONS = {   # our keyboard: /fk_pose with the fingertip resting on top of each key
    "p": [0.3469533113994126, 0.010143775622595701, -0.010003128903425973],
    "o": [0.36405435226863503, 0.021927450959049046, -0.007396726882837101],
    "l": [0.35467645554263066, 0.0419585218267298, -0.009674835713889662],
}
# For each value of the "keyboard" parameter: the key positions, and how far below a key's top
# to aim when pressing. Simulated keys travel 6 mm; real keys typically bottom out after about 4 mm.
KEYBOARDS = {
    "sim": (SIM_KEY_POSITIONS, 0.008),
    "real": (REAL_KEY_POSITIONS, 0.0055),
}

HOME_POSE = {
    "coordinates": [0.2659603329618999, 0.01930348180848848, 0.08971156030411485],
    "arm_state": [-4.571428571428571, 1.2747252747252746, 8.791208791208792, 86.15384615384616, 6.197802197802198, 1.6382252559726962]
}

HOVER_HEIGHT = 0.04   # metres above the key before and after pressing
MOVE_TIME = 1.5       # seconds for each of the four moves (default of the move_time parameter)
RATE_HZ = 20.0
FINGER_DOWN = [0.0, 0.0, -1.0]
GRIPPER_CLOSED = 0.0
MAX_ERROR = 0.005     # metres. A key whose IK solution is further off than this is not typed.
# Feedback on the joint angles: the servos stop a few degrees short of their command under load
# (gravity, the key's spring). While a waypoint is held, the remaining error is added to the command.
SETTLE_TIME = 0.8         # seconds to hold each waypoint so the correction can act
CORRECTION_GAIN = 0.15    # fraction of the remaining joint error added to the correction per step
MAX_CORRECTION = 4.0      # degrees per joint, so a target that cannot be reached is not pushed hard


class KeyboardTyper(Node):
    def __init__(self):
        super().__init__("keyboard_typer")
        self.arm_state = None
        self.waypoints = []      # joint-angle targets still to visit, each a list of 6 numbers
        self.start = None        # where the current move started
        self.steps_done = 0
        self.correction = np.zeros(5)   # degrees added to the 5 joint commands (see SETTLE_TIME)
        self.pending = []        # letters of the current word that are still to be typed
        self.declare_parameter("keyboard", "sim")
        self.declare_parameter("move_time", MOVE_TIME)
        keyboard = self.get_parameter("keyboard").value
        if keyboard not in KEYBOARDS:
            raise RuntimeError(f"keyboard must be one of {sorted(KEYBOARDS)}, got {keyboard!r}")
        self.key_positions, self.press_depth = KEYBOARDS[keyboard]
        self.move_time = self.get_parameter("move_time").value
        # TODO 1: create the publisher for "arm_commands" (self.publisher), subscribe to "key" (String,
        # self.on_key) and "arm_state" (Float64MultiArray, self.on_arm_state), and create a timer that
        # calls self.on_timer RATE_HZ times per second.
        ### Our code START
        self.publisher = self.create_publisher(Float64MultiArray, "arm_commands", 10)
        self.create_subscription(String, "key", self.on_key, 10)
        self.create_subscription(Float64MultiArray, "arm_state", self.on_arm_state, 10)
        self.create_timer(1.0 / RATE_HZ, self.on_timer)
        ### Our code END
        self.get_logger().info(f"KeyboardTyper started ({keyboard} keyboard). Keys: {sorted(self.key_positions)}")

    def on_arm_state(self, msg):
        self.arm_state = list(msg.data)

    def on_key(self, msg):
        # One message holds one letter ("p") or a whole word ("pool"); spaces are ignored.
        letters = [c for c in msg.data.lower() if not c.isspace()]
        if not letters or any(c not in self.key_positions for c in letters):
            self.get_logger().warn(f"Unknown key in {msg.data.strip()!r}. Known keys: {sorted(self.key_positions)}")
            return
        if self.waypoints or self.pending:
            self.get_logger().warn("Still busy with the previous key. Ignored.")
            return
        if self.arm_state is None:
            self.get_logger().warn("No arm_state received yet.")
            return
        self.pending = letters
        self.start_next_key()

    def start_next_key(self):
        """Plan the moves for the next letter in self.pending, starting from where the arm is now."""
        letter = self.pending.pop(0)
        # TODO 2: build the plan for this key. Compute three fingertip targets (above the key, pressed,
        # above again), solve inverse kinematics for each, starting each solve from the previous result,
        # and store the three results (5 angles + GRIPPER_CLOSED each) in self.waypoints. Also set
        # self.start to the current arm_state and self.steps_done to 0.
        ### Our code START
        # After the key, the fingertip also goes back to the home pose (above the keyboard).
        top = np.asarray(self.key_positions[letter], float)
        targets = [("above", top + [0.0, 0.0, HOVER_HEIGHT]),
                   ("press", top - [0.0, 0.0, self.press_depth]),
                   ("above", top + [0.0, 0.0, HOVER_HEIGHT]),
                   ("home", np.asarray(HOME_POSE["coordinates"], float))]
        angles, plan = self.arm_state, []
        for name, target in targets:
            angles, error = inverse_kinematics(target, FINGER_DOWN, angles)
            if error > MAX_ERROR:
                self.get_logger().warn(f"Key {letter!r} is out of reach ({name}: error {error * 1000:.0f} mm). "
                                       "Not moving.")
                self.pending = []   # do not type the rest of the word either
                return
            plan.append([*map(float, angles), GRIPPER_CLOSED])
        self.get_logger().info(f"Typing {letter!r}")
        self.waypoints = plan
        self.start = list(self.arm_state)
        self.steps_done = 0
        ### Our code END

    def on_timer(self):
        if not self.waypoints:
            return
        # TODO 3: publish one command that lies between self.start and self.waypoints[0], so that the
        # move takes MOVE_TIME seconds. When the waypoint is reached, remove it from the list and make
        # it the new self.start.
        ### Our code START
        self.steps_done += 1
        move_steps = max(1.0, self.move_time * RATE_HZ)
        fraction = min(1.0, self.steps_done / move_steps)
        start, target = np.asarray(self.start), np.asarray(self.waypoints[0])
        command = start + (target - start) * fraction
        if fraction >= 1.0:   # holding the waypoint: integrate the joint error that is left
            error = target[:5] - np.asarray(self.arm_state[:5])
            self.correction = np.clip(self.correction + CORRECTION_GAIN * error, -MAX_CORRECTION, MAX_CORRECTION)
        command[:5] += self.correction
        self.publisher.publish(Float64MultiArray(data=command.tolist()))
        if self.steps_done >= move_steps + SETTLE_TIME * RATE_HZ:
            self.start = self.waypoints.pop(0)
            self.steps_done = 0
            if not self.waypoints and self.pending:   # letter done and back home: next letter of the word
                self.start_next_key()
        ### Our code END


def main():
    rclpy.init()
    node = KeyboardTyper()
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
