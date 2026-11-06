"""A particle filter that estimates the robot's 2-D pose (x, y, heading) in a map of ArUco markers.

Plain NumPy, no ROS: the localization node feeds it, and you can test it on its own.

Every particle is one guess of the robot's pose, a row [x (m), y (m), theta (rad)] of the array
`particles` (N rows). Each has a weight: how well that guess explains what the camera saw. The filter
repeats three steps:

    predict    the robot moved (by odometry): move every particle the same way, plus some random
               noise, because odometry is not exact
    update     the camera saw markers: compare what each particle would expect to see with what was
               seen, and multiply its weight by how likely the measurement is from there
    resample   draw a new set of particles from the old one, in proportion to the weights: likely
               guesses are copied, unlikely ones disappear

The measurement of one marker is written in the robot's frame (x forward, y left), in 2-D:

    range    distance from the robot's centre to the marker (m)
    bearing  direction of the marker, counter-clockwise from straight ahead (rad)
    facing   the direction the marker's printed side faces, in the robot's frame (rad)
"""
import numpy as np

# A measurement error larger than this many standard deviations counts as this many. Without such a
# cap, a single bad detection (a wrong id, or a marker whose facing came out flipped) makes the right
# particles look impossible, and the filter jumps to wherever the bad detection happens to fit.
OUTLIER_CAP = 3.0


def wrap(angle):
    """Bring angles into [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def motion_update(particles, delta, noise, rng):
    """Predict: move every particle by the odometry step `delta`, with random noise.

    Args:
        particles: N x 3 array of poses [x, y, theta] in the map frame
        delta: [dx, dy, dtheta], how far the robot moved since the last call, in the robot's own frame
            at the start of the move (dx forward, dy left, in m; dtheta in rad)
        noise: [alpha_translation, alpha_rotation, sigma_xy_min, sigma_theta_min]. The noise added to each
            particle's step is normal with standard deviation
                alpha_translation * distance moved + sigma_xy_min        for dx and dy (m)
                alpha_rotation * |dtheta| + sigma_theta_min              for dtheta (rad)
        rng: a numpy random Generator (np.random.default_rng())
    Returns:
        the moved particles, N x 3, with theta wrapped to [-pi, pi)
    """
    particles = particles.copy()
    # OUR CODE START
    dx = delta[0]                 # forward move
    dy = delta[1]                 # sideways move
    dth = delta[2]                # turn

    dist = np.sqrt(dx*dx + dy*dy)                 # how far the robot moved
    sigma_xy = noise[0] * dist + noise[2]         # position noise
    sigma_th = noise[1] * abs(dth) + noise[3]     # turn noise

    for i in range(len(particles)):               # go through each particle one by one
        x = particles[i, 0]                       # this particle's x
        y = particles[i, 1]                       # this particle's y
        th = particles[i, 2]                      # this particle's heading

        my_dx = dx + rng.normal(0, sigma_xy)      # this particle's noisy forward move
        my_dy = dy + rng.normal(0, sigma_xy)      # this particle's noisy sideways move
        my_dth = dth + rng.normal(0, sigma_th)    # this particle's noisy turn

        x = x + np.cos(th) * my_dx - np.sin(th) * my_dy   # move in the particle's direction (x)
        y = y + np.sin(th) * my_dx + np.cos(th) * my_dy   # move in the particle's direction (y)
        th = th + my_dth                                  # turn the particle

        th = (th + np.pi) % (2 * np.pi) - np.pi   # keep the angle between -pi and pi

        particles[i, 0] = x                       # save the new x
        particles[i, 1] = y                       # save the new y
        particles[i, 2] = th                      # save the new heading
    # OUR CODE END
    return particles


def expected_measurement(particles, marker):
    """What each particle expects to measure for one marker of the map.

    Args:
        particles: N x 3 array of poses [x, y, theta] in the map frame
        marker: [x, y, facing] of the marker in the map frame (facing: the direction its printed side
            faces, in rad)
    Returns:
        three arrays of length N: range (m), bearing (rad) and facing (rad) of the marker as seen from
        each particle, in the particle's own frame
    """
    n = len(particles)
    expected_range, expected_bearing, expected_facing = np.zeros(n), np.zeros(n), np.zeros(n)
    # OUR CODE START
    x = particles[:, 0]          # x of every particle (whole column at once)
    y = particles[:, 1]          # y of every particle
    th = particles[:, 2]         # heading of every particle

    dx = marker[0] - x           # map-x distance from each particle to the marker
    dy = marker[1] - y           # map-y distance from each particle to the marker

    expected_range = np.hypot(dx, dy)                  # straight-line distance
    expected_bearing = wrap(np.arctan2(dy, dx) - th)   # direction to marker relative to particle heading
    expected_facing = wrap(marker[2] - th)             # marker facing seen from the particle
    # OUR CODE END
    return expected_range, expected_bearing, expected_facing


def measurement_log_likelihood(particles, measurements, marker_map, sigmas):
    """Update: how likely the measurements are from each particle, as a logarithm.

    Treat the measurements as independent, with normally distributed errors. For each measured marker
    that is in the map, compare the measured range, bearing and facing with the expected ones and add
        -0.5 * min(error / sigma, OUTLIER_CAP) ** 2
    for each of the three (the cap makes the filter robust against bad detections). Wrap angle errors
    to [-pi, pi) before dividing them by sigma.

    Args:
        particles: N x 3 array of poses
        measurements: list of (marker id, range, bearing, facing), in the robot's frame
        marker_map: {marker id: [x, y, facing]} in the map frame
        sigmas: [sigma_range_fraction, sigma_bearing, sigma_facing_per_metre]. The standard deviations of
            the range error and of the facing error grow with the distance, because a marker further
            away covers fewer pixels: sigma_range_fraction * measured range (m), and
            sigma_facing_per_metre * measured range (rad). sigma_bearing (rad) is the same at any distance.
    Returns:
        array of length N: the log-likelihood of each particle (0 for all if no marker is in the map)
    """
    log_likelihood = np.zeros(len(particles))
    # OUR CODE START
    for m in measurements:                         # each marker the camera saw
        marker_id, r, b, f = m                     # id, measured range, bearing, facing
        if marker_id not in marker_map:            # not in our map? ignore it
            continue

        marker = marker_map[marker_id]             # where this marker is on the map
        exp_r, exp_b, exp_f = expected_measurement(particles, marker)  # what each particle expected

        sigma_r = sigmas[0] * r                    # range noise grows with distance
        sigma_b = sigmas[1]                        # bearing noise is constant
        sigma_f = sigmas[2] * r                    # facing noise grows with distance

        err_r = abs(r - exp_r) / sigma_r           # range error in sigmas
        err_b = abs(wrap(b - exp_b)) / sigma_b     # bearing error (wrapped) in sigmas
        err_f = abs(wrap(f - exp_f)) / sigma_f     # facing error (wrapped) in sigmas

        log_likelihood -= 0.5 * np.minimum(err_r, OUTLIER_CAP) ** 2   # capped range penalty
        log_likelihood -= 0.5 * np.minimum(err_b, OUTLIER_CAP) ** 2   # capped bearing penalty
        log_likelihood -= 0.5 * np.minimum(err_f, OUTLIER_CAP) ** 2   # capped facing penalty
    # OUR CODE END
    return log_likelihood


def low_variance_resample(weights, rng):
    """Resample: choose N particle indices, each index i with probability weights[i].

    Use low-variance (systematic) resampling: one random number r in [0, 1/N), then pick the particles
    at the positions r, r + 1/N, r + 2/N, ... along the running sum of the weights. Particles with a large
    weight are picked several times, particles with a tiny weight usually not at all.

    Args:
        weights: array of length N, adding up to 1
        rng: a numpy random Generator
    Returns:
        integer array of length N: the indices of the chosen particles
    """
    n = len(weights)
    indices = np.arange(n)
    # OUR CODE START
    r = rng.uniform(0, 1.0 / n)            # one random start between 0 and 1/N
    positions = r + np.arange(n) / n       # N evenly spaced pointers

    cumulative = np.cumsum(weights)        # running sum of the weights (the "ruler")
    cumulative[-1] = 1.0                   # make the last value exactly 1

    indices = np.searchsorted(cumulative, positions)   # which particle each pointer lands on
    # OUR CODE END
    return indices


class ParticleFilter:
    """The filter itself: holds the particles and weights and runs the three steps.

    Args:
        num_particles: how many guesses to keep
        bounds: [x_min, x_max, y_min, y_max] of the area the robot can be in (m). Used to spread the
            particles at the start and to place the random particles.
        rng: a numpy random Generator (a new one if not given)
        recovery_rates: [slow, fast]. The filter keeps a slow and a fast running average of how well the
            particles explain the measurements. When the fast one drops below the slow one, the
            measurements have suddenly stopped fitting: the filter is probably wrong, and at the next
            resampling it replaces the fraction 1 - fast / slow of the particles by random ones. This is
            "augmented MCL" (Thrun, Burgard and Fox, Probabilistic Robotics, section 8.3.5). [0, 0] turns it off.
    """

    def __init__(self, num_particles, bounds, rng=None, recovery_rates=(0.0, 0.0)):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.bounds = bounds
        self.particles = np.zeros((num_particles, 3))
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self.recovery_rates = recovery_rates
        self.fit_slow = self.fit_fast = None   # running averages of the measurement fit
        self.initialize_uniform()

    def initialize_uniform(self):
        """Spread the particles evenly over the whole area, with random headings: "I could be anywhere"."""
        self.particles = self.random_particles(len(self.particles))
        self.weights[:] = 1.0 / len(self.particles)

    def initialize_around(self, pose, sigma_xy, sigma_theta):
        """Put the particles around a known pose: "I am about here"."""
        n = len(self.particles)
        self.particles = np.column_stack([self.rng.normal(pose[0], sigma_xy, n), self.rng.normal(pose[1], sigma_xy, n),
                                          wrap(self.rng.normal(pose[2], sigma_theta, n))])
        self.weights[:] = 1.0 / n

    def random_particles(self, n):
        x_min, x_max, y_min, y_max = self.bounds
        return np.column_stack([self.rng.uniform(x_min, x_max, n), self.rng.uniform(y_min, y_max, n),
                                self.rng.uniform(-np.pi, np.pi, n)])

    def predict(self, delta, noise):
        self.particles = motion_update(self.particles, delta, noise, self.rng)

    def update(self, measurements, marker_map, sigmas):
        """Multiply the weights by the likelihood of the measurements. Returns False if nothing was used."""
        if not any(marker_id in marker_map for marker_id, *_ in measurements):
            return False
        log_likelihood = measurement_log_likelihood(self.particles, measurements, marker_map, sigmas)
        # How well the particles explain this measurement: the weighted average likelihood per marker
        # (per marker, so that seeing more markers does not by itself look like a worse fit).
        used = sum(marker_id in marker_map for marker_id, *_ in measurements)
        fit = self.weights @ np.exp(log_likelihood / used)
        slow, fast = self.recovery_rates
        if self.fit_slow is None:
            self.fit_slow = self.fit_fast = fit
        self.fit_slow += slow * (fit - self.fit_slow)
        self.fit_fast += fast * (fit - self.fit_fast)
        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        # Subtract the largest value before exponentiating, so that the numbers do not all round to 0.
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()
        return True

    def effective_sample_size(self):
        """About how many particles really count: N if all weights are equal, 1 if one has all the weight."""
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self, noise_xy, noise_theta, random_fraction):
        """Draw a new set of particles in proportion to the weights.

        After copying, every particle is shifted by a little random noise (noise_xy in m, noise_theta in
        rad), so that copies of the same particle spread out and explore the area around it. A fraction
        random_fraction of the particles is replaced by random ones anywhere in the area, so that the
        filter can recover if it is ever completely wrong (for example if someone picks the robot up and
        carries it elsewhere). When the measurements have recently stopped fitting (see recovery_rates),
        that fraction is raised.
        """
        n = len(self.particles)
        self.particles = self.particles[low_variance_resample(self.weights, self.rng)]
        self.particles[:, :2] += self.rng.normal(0.0, noise_xy, (n, 2))
        self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0.0, noise_theta, n))
        if self.fit_slow:
            random_fraction = max(random_fraction, 1.0 - self.fit_fast / self.fit_slow)
        n_random = int(round(random_fraction * n))
        if n_random:
            self.particles[self.rng.choice(n, n_random, replace=False)] = self.random_particles(n_random)
        self.weights[:] = 1.0 / n

    def best(self):
        """The particle with the largest weight."""
        return self.particles[np.argmax(self.weights)].copy()

    def mean(self):
        """The weighted mean of the particles (the heading is averaged as a direction)."""
        x, y = self.weights @ self.particles[:, 0], self.weights @ self.particles[:, 1]
        theta = np.arctan2(self.weights @ np.sin(self.particles[:, 2]), self.weights @ np.cos(self.particles[:, 2]))
        return np.array([x, y, theta])

    def covariance(self):
        """3 x 3 weighted covariance of the particles around their mean (x, y in m; theta in rad)."""
        difference = self.particles - self.mean()
        difference[:, 2] = wrap(difference[:, 2])
        return (difference * self.weights[:, None]).T @ difference