"""A small MuJoCo simulator of the LeKiwi robot (omni-wheel base + SO-101 style arm).

The simulator speaks the same units as the real robot does through LeRobot:

    arm joints     degrees, 0 = the middle of each joint's range
    gripper        percent, 0 = closed, 100 = open
    base velocity  forward m/s, left m/s, counter-clockwise deg/s

Typical use:

    from lekiwi_sim import LeKiwiSim

    sim = LeKiwiSim()
    sim.set_command(arm=[0, 0, 0, 0, 0, 50], base=[0.1, 0.0, 0.0])
    for _ in range(100):
        sim.step()
    print(sim.get_state()["base_pose"])
"""

import atexit
import math
import os
import time
from pathlib import Path

# Without a monitor (for example over SSH) MuJoCo needs the EGL backend to draw camera images.
# Set MUJOCO_GL yourself before starting Python to override it.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] == "egl":
    # mujoco sets this itself, but only if MUJOCO_GL was already set when mujoco was first imported.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "LeKiwi-sim" / "mjcf_lcmm_robot.xml"

# ---------------------------------------------------------------------------------------------
# Robot description: names in the MuJoCo model and how they map to the real robot's units.
# ---------------------------------------------------------------------------------------------

# Command/state order of the arm, using the real robot's joint names.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# The same joints under the names the MuJoCo model uses.
MODEL_ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
MODEL_WHEEL_ACTUATORS = ("drive_motor_1", "drive_motor_2", "drive_motor_3")
MODEL_BASE_BODY = "base_plate_layer1-v5-1"
MODEL_FIXED_JAW_BODY = "Wrist_Roll_08c-v1"
MODEL_MOVING_JAW_BODY = "Moving_Jaw_08d-v1"

# model_angle_deg = ARM_SIGN * real_angle_deg + ARM_OFFSET_DEG, for the five rotary arm joints.
# The model's joints do not share the real robot's zero pose or turning direction. These values
# come from the reference project, which matched the model against a real arm joint by joint.
ARM_SIGN = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
ARM_OFFSET_DEG = np.array([0.0, -90.0, 90.0, 80.0, 270.0])
# The gripper is a percentage on the real robot. 0 % and 100 % map to these jaw angles (radians).
JAW_CLOSED_RAD, JAW_OPEN_RAD = -0.174533, 1.745329

# In the model's base frame the robot's front points along +y and its left side along -x.
FORWARD_IN_BASE = np.array([0.0, 1.0, 0.0])
LEFT_IN_BASE = np.array([-1.0, 0.0, 0.0])
# Distance from a wheel's axle to the floor: roller circle radius 5.08 cm + roller radius 1.2 cm.
WHEEL_RADIUS = 0.0628
# The real wheel servos top out at 3000 encoder steps per second (4096 steps per turn).
MAX_WHEEL_SPEED = 3000.0 * 2.0 * math.pi / 4096.0  # rad/s
# Where the "gripper" marker sits in the fixed jaw's own frame (metres): between the fingertips.
GRIPPER_SITE_POS = [0.008, 0.0, 0.090]
# Gripper motor of the simulated robot (see _build_model for why it differs from the reference model).
GRIPPER_STIFFNESS = 20.0   # N m per radian of error between target and actual jaw angle
GRIPPER_MAX_TORQUE = 0.5   # N m, about 8 N of squeeze at the fingertips
# Contact settings for the finger pads and the cubes. MuJoCo's contacts are slightly soft by
# default; these values make them stiff enough that a squeezed cube is not pushed into the fingers.
GRASP_SOLREF = [0.004, 1.0]
GRASP_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]
# A servo cannot jump to a new angle. The simulated arm follows its targets at up to this speed.
MAX_ARM_SPEED = 180.0  # deg/s

# ---------------------------------------------------------------------------------------------
# Scene: what is placed around the robot. The robot starts at the origin, facing +x.
# ---------------------------------------------------------------------------------------------

CUBE_SIZE = 0.03  # edge length in metres
# name: (x, y, colour). Each cube starts resting on the platform below it.
CUBES = {
    "red_cube": (0.30, 0.06, (0.85, 0.15, 0.15, 1.0)),
    "green_cube": (0.30, -0.06, (0.15, 0.65, 0.20, 1.0)),
    "blue_cube": (0.36, 0.00, (0.15, 0.35, 0.90, 1.0)),
}
# name: (centre x, centre y, size x, size y, height, colour). Static boxes standing on the floor.
PLATFORMS = {
    "table": (0.34, 0.0, 0.16, 0.30, 0.04, (0.75, 0.70, 0.60, 1.0)),
    "shelf_left": (0.10, 0.34, 0.20, 0.16, 0.08, (0.55, 0.60, 0.70, 1.0)),
    "shelf_right": (0.10, -0.34, 0.20, 0.16, 0.12, (0.55, 0.60, 0.70, 1.0)),
}

# Collision bits. The reference model switches collisions off for almost every robot part and
# uses bit 1 for wheels/floor and bit 2 for a few spheres that keep the arm above the floor.
# We add bit 4 for the two gripper jaws so that they can touch cubes and platforms.
BIT_WORLD, BIT_ARM_PROXY, BIT_JAW = 1, 2, 4


def _look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w, x, y, z) for a MuJoCo camera at `position` looking at `target`.

    `up` is the direction that should point to the top of the image.
    """
    position, target = np.asarray(position, float), np.asarray(target, float)
    back = position - target
    back /= np.linalg.norm(back)  # a MuJoCo camera looks along its own -z axis
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, back]).flatten())
    return quat


def _make_renderer(model, width, height):
    """Create an off-screen renderer that also works without a monitor.

    MuJoCo picks its drawing backend when `mujoco` is first imported. If your script imported
    mujoco before this module, it picked the windowed backend, which fails over SSH. Here we
    point the renderer at the EGL backend explicitly, so the import order does not matter.
    """
    if os.environ.get("MUJOCO_GL") == "egl":
        try:
            import mujoco.egl
            from mujoco.rendering.classic import gl_context
            gl_context.GLContext = mujoco.egl.GLContext
        except ImportError:  # other MuJoCo versions: fall back to whatever mujoco chose
            pass
    return mujoco.Renderer(model, height, width)


class LeKiwiSim:
    """MuJoCo simulation of LeKiwi with a few cubes and platforms.

    Args:
        control_dt: seconds of simulated time that one call to step() advances.
        model_path: MJCF file of the robot. Defaults to the model in models/LeKiwi-sim.
        add_scene: set to False to load the robot alone, without cubes and platforms.
    """

    def __init__(self, control_dt=0.02, model_path=None, add_scene=True):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Robot model not found at {model_path}. "
                "In the repository folder run: git submodule update --init"
            )
        self.model = self._build_model(model_path, add_scene)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self._substeps = max(1, round(control_dt / self.model.opt.timestep))

        m = self.model
        name2id = mujoco.mj_name2id
        self._base_body = name2id(m, mujoco.mjtObj.mjOBJ_BODY, MODEL_BASE_BODY)
        self._gripper_site = name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        arm_joint_ids = [name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in MODEL_ARM_JOINTS]
        self._arm_qpos = np.array([m.jnt_qposadr[j] for j in arm_joint_ids])
        self._arm_qvel = np.array([m.jnt_dofadr[j] for j in arm_joint_ids])
        self._arm_ctrl = np.array([name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in MODEL_ARM_JOINTS])
        self._wheel_ctrl = np.array([name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in MODEL_WHEEL_ACTUATORS])
        self._wheel_qvel = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in self._wheel_ctrl])
        self._object_bodies = {
            n: name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in CUBES if name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
        }
        self.camera_names = ["front", "wrist", "overview", "top"]

        self._renderers = {}
        self._viewer = None
        self._wall_start = None
        self._arm_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self._base_command = np.zeros(3)
        self.reset()
        self._wheel_geometry = self._measure_wheels()

    # -----------------------------------------------------------------------------------------
    # Building the model
    # -----------------------------------------------------------------------------------------

    @staticmethod
    def _build_model(model_path, add_scene):
        """Load the robot MJCF and add our own parts to it in memory (the file is not changed)."""
        spec = mujoco.MjSpec.from_file(str(model_path))
        box = mujoco.mjtGeom.mjGEOM_BOX

        # Turn the robot by -90 degrees so that its front (+y in its own frame) points along world +x.
        base = spec.body(MODEL_BASE_BODY)
        base.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        base.quat = [math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)]

        # Let the two gripper fingers touch things. The reference model has no collision shapes on
        # the gripper, so we add one invisible box ("pad") along the inside of each finger.
        # Each entry: body, pad centre and pad half-sizes, in that body's own frame (metres).
        pads = ((MODEL_FIXED_JAW_BODY, [-0.018, 0.0, 0.075], [0.006, 0.009, 0.030]),
                (MODEL_MOVING_JAW_BODY, [-0.003, -0.052, 0.019], [0.006, 0.030, 0.009]))
        for body_name, pos, size in pads:
            spec.body(body_name).add_geom(name=body_name + "_pad", type=box, pos=pos, size=size, group=3,
                                          rgba=[0.1, 0.8, 0.1, 0.4], contype=BIT_JAW, conaffinity=0,
                                          friction=[2.0, 0.05, 0.001], condim=4, mass=0.0,
                                          solref=GRASP_SOLREF, solimp=GRASP_SOLIMP)
        # Mark the point between the fingertips, and give the gripper a realistic closing force.
        spec.body(MODEL_FIXED_JAW_BODY).add_site(name="gripper", pos=GRIPPER_SITE_POS, size=[0.004, 0.004, 0.004],
                                                 rgba=[1.0, 0.9, 0.0, 1.0])
        # In the reference model the gripper motor can push with 80 N m and its joint is given a
        # very large inertia. That is strong enough to drive the fingers straight through a small
        # cube. We replace it with a weaker, lighter motor that closes in about a second and then
        # squeezes with a few newtons.
        jaw_joint = spec.joint("Jaw")
        jaw_joint.armature = 0.005
        # Recent MuJoCo versions store damping as three polynomial coefficients, older ones as one number.
        jaw_joint.damping = [0.4, 0.0, 0.0] if np.ndim(jaw_joint.damping) else 0.4
        jaw = spec.actuator("Jaw")
        gain, bias = np.zeros(len(jaw.gainprm)), np.zeros(len(jaw.biasprm))
        gain[0], bias[1] = GRIPPER_STIFFNESS, -GRIPPER_STIFFNESS
        jaw.gainprm, jaw.biasprm = gain, bias
        jaw.forcerange = [-GRIPPER_MAX_TORQUE, GRIPPER_MAX_TORQUE]

        # The reference model ships with one block on the floor. We turn it into one of our cubes.
        block = spec.body("target_block")
        world = spec.worldbody
        if add_scene:
            touches = BIT_WORLD | BIT_ARM_PROXY | BIT_JAW
            for name, (x, y, sx, sy, height, rgba) in PLATFORMS.items():
                world.add_geom(name=name, type=box, pos=[x, y, height / 2], size=[sx / 2, sy / 2, height / 2],
                               rgba=rgba, contype=BIT_WORLD, conaffinity=touches, friction=[1.0, 0.005, 0.0001])
            half = CUBE_SIZE / 2
            for i, (name, (x, y, rgba)) in enumerate(CUBES.items()):
                height = max(
                    [h for (px, py, sx, sy, h, _) in PLATFORMS.values()
                     if abs(x - px) <= sx / 2 and abs(y - py) <= sy / 2] or [0.0])
                if i == 0:
                    body, geom = block, block.geoms[0]
                    body.name, geom.name = name, name + "_geom"
                    body.pos = [x, y, height + half]
                else:
                    body = world.add_body(name=name, pos=[x, y, height + half])
                    body.add_freejoint(name=name + "_free")
                    geom = body.add_geom(name=name + "_geom", type=box)
                geom.size = [half, half, half]
                geom.rgba = rgba
                geom.mass = 0.02
                geom.contype = BIT_WORLD
                geom.conaffinity = BIT_WORLD | BIT_JAW
                geom.friction = [2.0, 0.05, 0.001]
                geom.condim = 4
                geom.solref = GRASP_SOLREF
                geom.solimp = GRASP_SOLIMP
        else:
            block.pos = [0.0, 3.5, 0.025]  # park the built-in block far away from the robot

        # Cameras. The reference model's own camera ("front_camera") points into the robot, so we
        # add our own: two on the robot (front of the base, and on the gripper) and two fixed ones.
        base.add_camera(name="front", pos=[0.0, 0.125, 0.03], fovy=70,
                        quat=_look_at([0.0, 0.125, 0.03], [0.0, 1.0, 0.03]))
        spec.body(MODEL_FIXED_JAW_BODY).add_camera(
            name="wrist", pos=[-0.065, 0.0, 0.02], fovy=75,
            quat=_look_at([-0.065, 0.0, 0.02], [0.02, 0.0, 0.13], up=(1.0, 0.0, 0.0)))
        world.add_camera(name="overview", pos=[-0.55, -0.75, 0.65], quat=_look_at([-0.55, -0.75, 0.65], [0.25, 0.0, 0.10]))
        world.add_camera(name="top", pos=[0.25, 0.0, 1.3], quat=[math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)])
        world.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.2, 0.2, -1.0])

        # Allow camera images up to 1280 x 960 pixels (MuJoCo's default limit is 640 x 480).
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        return spec.compile()

    def _measure_wheels(self):
        """For each wheel: (hub position, rolling direction) in the base frame, read from the model."""
        rot = self.data.xmat[self._base_body].reshape(3, 3)
        origin = self.data.xpos[self._base_body]
        wheels = []
        for actuator in self._wheel_ctrl:
            joint = self.model.actuator_trnid[actuator, 0]
            hub = rot.T @ (self.data.xanchor[joint] - origin)
            axle = rot.T @ self.data.xaxis[joint]
            # A wheel turning at +w about its axle moves its hub along (axle x up) * radius * w.
            wheels.append((hub, np.cross(axle, [0.0, 0.0, 1.0])))
        return wheels

    # -----------------------------------------------------------------------------------------
    # Unit conversions between the real robot's convention and the model
    # -----------------------------------------------------------------------------------------

    @staticmethod
    def _arm_to_model(arm):
        """[5 joint angles in degrees, gripper in %] -> the model's six joint angles in radians."""
        arm = np.asarray(arm, float)
        rad = np.radians(ARM_SIGN * arm[:5] + ARM_OFFSET_DEG)
        percent = min(100.0, max(0.0, arm[5]))
        jaw = JAW_CLOSED_RAD + percent / 100.0 * (JAW_OPEN_RAD - JAW_CLOSED_RAD)
        return np.append(rad, jaw)

    @staticmethod
    def _arm_from_model(rad):
        """Inverse of _arm_to_model."""
        deg = ARM_SIGN * (np.degrees(rad[:5]) - ARM_OFFSET_DEG)
        percent = (rad[5] - JAW_CLOSED_RAD) / (JAW_OPEN_RAD - JAW_CLOSED_RAD) * 100.0
        return np.append(deg, percent)

    def _base_to_wheels(self, base):
        """[forward m/s, left m/s, counter-clockwise deg/s] -> three wheel speeds in rad/s.

        Each wheel hub moves at v + omega x r. An omni wheel only has to follow the part of that
        motion that lies along its rolling direction; its rollers let it slide sideways.
        """
        velocity = base[0] * FORWARD_IN_BASE + base[1] * LEFT_IN_BASE
        spin = np.array([0.0, 0.0, math.radians(base[2])])
        speeds = np.array([(velocity + np.cross(spin, hub)) @ rolling / WHEEL_RADIUS
                           for hub, rolling in self._wheel_geometry])
        fastest = np.abs(speeds).max()
        if fastest > MAX_WHEEL_SPEED:  # like the real robot: slow all wheels down together
            speeds *= MAX_WHEEL_SPEED / fastest
        return speeds

    # -----------------------------------------------------------------------------------------
    # The interface you program against
    # -----------------------------------------------------------------------------------------

    def reset(self):
        """Put the robot and all objects back at their start poses and zero the commands."""
        mujoco.mj_resetData(self.model, self.data)
        self._arm_command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        self._base_command = np.zeros(3)
        start = self._arm_to_model(self._arm_command)
        self.data.qpos[self._arm_qpos] = start
        self.data.ctrl[self._arm_ctrl] = start
        mujoco.mj_forward(self.model, self.data)
        self._wall_start = None
        if self._viewer is not None:
            self._viewer.sync()

    def set_command(self, arm=None, base=None):
        """Set the targets the robot follows from now on.

        Args:
            arm: six numbers [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] in
                degrees and [gripper] in percent. Position targets: the arm moves there (at up
                to MAX_ARM_SPEED) and holds.
            base: three numbers [forward m/s, left m/s, counter-clockwise deg/s]. A velocity:
                the base keeps moving like this until you send a new command.

        Leave an argument out (or pass None) to keep its previous value.
        """
        if arm is not None:
            arm = np.asarray(arm, float)
            if arm.shape != (6,):
                raise ValueError(f"arm must have 6 values, got {arm.shape}")
            self._arm_command = arm.copy()
        if base is not None:
            base = np.asarray(base, float)
            if base.shape != (3,):
                raise ValueError(f"base must have 3 values, got {base.shape}")
            self._base_command = base.copy()

    def get_command(self):
        """Return the current targets as {"arm": 6 values, "base": 3 values}."""
        return {"arm": self._arm_command.copy(), "base": self._base_command.copy()}

    def step(self, n=1, realtime=False):
        """Advance the simulation by n control periods (n * control_dt seconds).

        With realtime=True the call also waits, so that simulated time does not run ahead of the
        clock on the wall. Use it together with the viewer.
        """
        lo, hi = self.model.actuator_ctrlrange[self._arm_ctrl].T
        max_change = math.radians(MAX_ARM_SPEED) * self.control_dt
        for _ in range(n):
            target = np.clip(self._arm_to_model(self._arm_command), lo, hi)
            current = self.data.ctrl[self._arm_ctrl]
            self.data.ctrl[self._arm_ctrl] = current + np.clip(target - current, -max_change, max_change)
            self.data.ctrl[self._wheel_ctrl] = self._base_to_wheels(self._base_command)
            for _ in range(self._substeps):
                mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
            if realtime:
                if self._wall_start is None:
                    self._wall_start = time.perf_counter() - self.data.time
                ahead = self.data.time - (time.perf_counter() - self._wall_start)
                if ahead > 0:
                    time.sleep(ahead)

    def get_state(self):
        """Return the current state of the robot and the objects as a dictionary.

        Keys:
            time            simulated seconds since the last reset
            arm_position    6 values: 5 joint angles (deg) and the gripper (%)
            arm_velocity    6 values: deg/s and %/s
            base_pose       [x (m), y (m), heading (deg)] in the world frame
            base_velocity   [forward (m/s), left (m/s), counter-clockwise (deg/s)] in the robot's frame
            wheel_velocity  3 wheel speeds (rad/s)
            gripper_position  [x, y, z] (m) of the point between the fingertips, in the world frame
            objects         {name: {"position": [x, y, z], "quaternion": [w, x, y, z]}}
        """
        d = self.data
        arm_rad = d.qpos[self._arm_qpos]
        scale = np.append(ARM_SIGN, 100.0 / math.degrees(JAW_OPEN_RAD - JAW_CLOSED_RAD))
        rot = d.xmat[self._base_body].reshape(3, 3)
        forward = rot @ FORWARD_IN_BASE
        velocity = np.zeros(6)  # [angular (3), linear (3)] in the base's own frame
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self._base_body, velocity, 1)
        return {
            "time": float(d.time),
            "arm_position": self._arm_from_model(arm_rad),
            "arm_velocity": np.degrees(d.qvel[self._arm_qvel]) * scale,
            "base_pose": np.array([d.xpos[self._base_body][0], d.xpos[self._base_body][1],
                                   math.degrees(math.atan2(forward[1], forward[0]))]),
            "base_velocity": np.array([velocity[3:] @ FORWARD_IN_BASE, velocity[3:] @ LEFT_IN_BASE,
                                       math.degrees(velocity[2])]),
            "wheel_velocity": d.qvel[self._wheel_qvel].copy(),
            "gripper_position": d.site_xpos[self._gripper_site].copy(),
            "objects": {name: {"position": d.xpos[body].copy(), "quaternion": d.xquat[body].copy()}
                        for name, body in self._object_bodies.items()},
        }

    def get_image(self, camera="overview", width=320, height=240):
        """Return what a camera sees as a (height, width, 3) array of 8-bit RGB values.

        camera is one of sim.camera_names: "front" (on the base, looking ahead), "wrist" (on the
        gripper, looking at the fingertips), "overview" or "top" (fixed in the room).
        """
        if camera not in self.camera_names:
            raise ValueError(f"Unknown camera {camera!r}. Choose one of {self.camera_names}")
        if (width, height) not in self._renderers:
            first = not self._renderers
            self._renderers[(width, height)] = _make_renderer(self.model, width, height)
            if first:
                # Free the renderers before Python shuts down. Registered here, after the first
                # renderer exists, so that it runs before MuJoCo's own clean-up of the EGL backend.
                atexit.register(self.close)
        renderer = self._renderers[(width, height)]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def open_viewer(self):
        """Open MuJoCo's interactive 3-D window. Needs a monitor (it does not work over SSH)."""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def viewer_is_open(self):
        """True while the 3-D window opened by open_viewer() has not been closed."""
        return self._viewer is not None and self._viewer.is_running()

    def close(self):
        """Close the viewer and free the camera renderers."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}

