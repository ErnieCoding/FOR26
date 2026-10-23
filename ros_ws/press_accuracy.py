"""PressAccuracy: where did the fingertip really go on each key press, compared with the target?

Run it next to keyboard_typer (passive: it only subscribes), type some keys, then press Ctrl+C.

Subscribes:
    key           the letters sent to the typer, in order
    arm_commands  what the typer asked for
    arm_state     where the arm really is
For every press (the fingertip going below the key tops) it reports, using your forward kinematics:
    miss          sideways distance between the key's press target and the fingertip at its lowest point
    depth         how far below the press target the fingertip went (negative = it stopped above it)
    servo errors  commanded minus actual angle of each joint at that moment

Usage:  python press_accuracy.py real        (or: sim)
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from arm_control.kinematics import forward_kinematics
from arm_control.keyboard_typer import KEYBOARDS

JOINTS = ["pan", "lift", "elbow", "wflex", "wroll"]


class PressAccuracy(Node):
    def __init__(self):
        super().__init__("press_accuracy")
        self.command = None
        self.rows = []   # time, command (5 angles), state (5 angles)
        self.letters = []
        self.create_subscription(String, "key", lambda msg: self.letters.append(msg.data.strip().lower()), 10)
        self.create_subscription(Float64MultiArray, "arm_commands", self.on_command, 10)
        self.create_subscription(Float64MultiArray, "arm_state", self.on_state, 10)
        self.get_logger().info("Recording. Type some keys, then press Ctrl+C for the report.")

    def on_command(self, msg):
        self.command = list(msg.data)[:5]

    def on_state(self, msg):
        if self.command is not None:
            self.rows.append([time.monotonic(), *self.command, *list(msg.data)[:5]])


def report(rows, letters, keyboard):
    keys, depth = KEYBOARDS[keyboard]
    if len(rows) < 2:
        print("Nothing recorded.")
        return
    rows = np.array(rows)
    cmd, state = rows[:, 1:6], rows[:, 6:11]
    act_tip = np.array([forward_kinematics(s)[0] for s in state])
    letters = [k for k in letters if k in keys]
    # A press is where the real fingertip goes below the lowest key top.
    below = act_tip[:, 2] < min(k[2] for k in keys.values()) - 0.001
    edges = np.flatnonzero(np.diff(below.astype(int)))
    starts = sorted([i + 1 for i in edges if not below[i]] + ([0] if below[0] else []))
    print(f"\n{'key':3s} {'miss':>8s} {'dx':>6s} {'dy':>6s} {'depth':>7s}   servo errors (cmd - actual, deg):  "
          + " ".join(f"{j:>6s}" for j in JOINTS))
    for key, s in zip(letters, starts):
        e = s
        while e < len(below) and below[e]:
            e += 1
        low = s + int(np.argmin(act_tip[s:e, 2]))
        d = (act_tip[low] - (np.array(keys[key]) - [0, 0, depth])) * 1000
        err = cmd[low] - state[low]
        print(f"{key:3s} {np.hypot(d[0], d[1]):6.1f}mm {d[0]:6.1f} {d[1]:6.1f} {-d[2]:5.1f}mm   "
              + " " * 33 + " ".join(f"{x:6.1f}" for x in err))
    print("dx: + = further from the robot, dy: + = to the robot's left. Model numbers (from your FK).")


def main():
    keyboard = sys.argv[1] if len(sys.argv) > 1 else "real"
    rclpy.init()
    node = PressAccuracy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        report(node.rows, node.letters, keyboard)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
