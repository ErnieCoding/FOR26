# ROS

In this lab you learn the basics of **ROS 2** by using it. You start with the ideas, build the course's ROS packages, write your own two nodes, and finish by moving the simulated robot's arm with the real leader arm.

```
 leader arm  ──►  leader_node  ──/arm_commands──►  sim_node (MuJoCo simulator)
 (your hand)                                            │
                                                        └──/arm_state──►  your follower node
```

Do the [Setup guide](../Setup/README.md) and the [Simulation lab](../Simulate/README.md) first. Then follow these pages in order.

| # | Page | What you do |
| --- | --- | --- |
| 1 | [ROS concepts](01_ros_concepts.md) | Learn what nodes, topics and messages are |
| 2 | [Build and run](02_build_and_run.md) | Build the workspace, start the simulator node, and inspect it with the `ros2` tools |
| 3 | [Your first nodes](03_your_first_nodes.md) | **Exercise:** write a leader (publisher) and a follower (subscriber) |
| 4 | [Teleoperation in simulation](04_teleop_in_sim.md) | Move the simulated arm with the real leader arm |

## What is in the workspace

The ROS code is in `~/FOR26/ros_ws/src`, in two packages.

| Package | Who writes it | Contents |
| --- | --- | --- |
| `lekiwi_ros` | provided | `sim_node` (the simulator as a ROS node) and `leader_node` (reads the real leader arm) |
| `hello_ros` | **you** | `leader` and `follower`, your first two nodes, with gaps for you to fill in |

Your TA has complete versions of your two nodes. Try first, ask later.

## The topics used in this lab

| Topic | Message type | Content |
| --- | --- | --- |
| `/arm_commands` | `std_msgs/Float64MultiArray` | 6 numbers: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` in degrees, and the gripper in percent |
| `/base_commands` | `std_msgs/Float64MultiArray` | 3 numbers: forward m/s, left m/s, turn deg/s |
| `/arm_state` | `std_msgs/Float64MultiArray` | 6 numbers, same layout as `/arm_commands`: where the arm really is |
| `/base_pose` | `std_msgs/Float64MultiArray` | 3 numbers: x (m), y (m), heading (deg) |
| `/camera/image` | `sensor_msgs/Image` | camera pictures, only if switched on |

These are the same numbers and units as in the [simulator class](../Simulate/03_the_simulator_class.md).
