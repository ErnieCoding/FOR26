"""Rebuild the leader arm's calibration file from the values stored inside its servos.

Usage: python recover_calibration.py <port> [name]

A calibration is stored in two places: inside the servos, and in a file on the computer that
LeRobot reads. If the arm was calibrated on a different computer, the file is missing here.
This program reads the values from the servos and writes the file. It does not move anything.
"""
import sys

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    port = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "leader"

    leader = SO101Leader(SO101LeaderConfig(port=port, id=name))
    leader.bus.connect()
    leader.calibration = leader.bus.read_calibration()
    print("Calibration stored in the servos:")
    for joint, c in leader.calibration.items():
        print(f"  {joint:14s} id {c.id}  homing_offset {c.homing_offset:6d}  range {c.range_min:5d} to {c.range_max:5d}")
    leader._save_calibration()
    leader.bus.disconnect()
    print("Saved to", leader.calibration_fpath)

    # Connect the normal way to check that LeRobot accepts the file, and read the arm once.
    leader = SO101Leader(SO101LeaderConfig(port=port, id=name))
    leader.connect(calibrate=False)
    print("File matches the servos:", leader.is_calibrated)
    print("Leader pose now:", {key.removesuffix(".pos"): round(value, 1) for key, value in leader.get_action().items()})
    leader.disconnect()


if __name__ == "__main__":
    main()
