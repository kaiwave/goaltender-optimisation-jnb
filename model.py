import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.optimize import minimize_scalar
from typing import Tuple

# ---------------------------------------
# CONSTANTS, CONSTRAINTS & PARAMETERS
# ---------------------------------------
# Net and crease dimensions in metres
W_NET: float = 1.83
H_NET: float = 1.22
R_CREASE: float = 1.83

# Goalie dimensions in metres
W_GOALIE: float = 1.15
H_GOALIE: float = 0.95

# Velocities and Reaction Times (ms^-1 and s)
V_PASS: float = 25
V_SLIDE: float = 3.5
T_REACT: float = 0.2

# Stochastic Parameters (Standard Deviations in meters)
SIGMA_X_DEFAULT: float = 0.40  # Lateral stickhandling release variance
SIGMA_Y_DEFAULT: float = 0.30  # Longitudinal stride release variance
N_SAMPLES: int = 1000  # Number of samples for stochastic simulation
RISK_THRESHOLD: float = 0.05  # Threshold for acceptable recovery risk

# ---------------------------------------
# CORE ENGINE - GEOMETRY AND PROJECTION
# ---------------------------------------
def get_geometry(
    x_p: float, 
    y_p: float
) -> tuple[float, float]:
  # Calc theta and distance from the point to the net
  theta = np.arctan2(x_p, y_p)
  distance_Ds = np.sqrt(x_p**2 + y_p**2) 

  return theta, distance_Ds

def get_net_apparent(
    theta: float
) -> float:
  
  # Calc apparent area of the net from the angle theta
  apparent_area = W_NET * H_NET * np.cos(theta)

  return apparent_area

def get_goalie_vertices(
    d_g: float, 
    theta: float
) -> np.ndarray:
  
  # Calc apparent width and height of the goalie from the angle theta
  dg_sin_theta = d_g * np.sin(theta)
  dg_cos_theta = d_g * np.cos(theta)
  wg_sin_theta = (W_GOALIE / 2) * np.sin(theta)
  wg_cos_theta = (W_GOALIE / 2) * np.cos(theta)

  x1 = dg_sin_theta - wg_cos_theta
  y1 = dg_cos_theta + wg_sin_theta
  v1 = np.array([x1, y1, 0])

  x2 = dg_sin_theta + wg_cos_theta
  y2 = dg_cos_theta - wg_sin_theta
  v2 = np.array([x2, y2, 0])

  # v3, v4 are just the same as v1, v2 but with the height of the goalie added
  v3 = np.array([x1, y1, H_GOALIE])
  v4 = np.array([x2, y2, H_GOALIE])

  goalie_vertices = np.array([v1, v2, v3, v4])
  return goalie_vertices

def puck_persp_proj(
    goalie_vertices: np.ndarray, 
    x_p: float, 
    y_p: float
) -> np.ndarray:
  
  # Project the goalie vertices onto the goal plane
  x_v = goalie_vertices[:, 0]
  y_v = goalie_vertices[:, 1]
  z_v = goalie_vertices[:, 2] # Take goalie vertices since they are already sorted

  den = np.where(np.abs(y_p - y_v) < 1e-6, 1e-6, y_p - y_v) # Check for divide by zero
  t = y_p / den

  x_proj = x_p - t * (x_p - x_v)
  z_proj = t * z_v # Arithmetic from the projection formula in 2.2.4

  return np.column_stack((x_proj, z_proj)) # Returns a 2D array of size (4, 2) with the projected coordinates

def exposed_area_eff(
    x_p: float, 
    y_p: float, 
    d_g: float, 
    theta: float = None
) -> tuple[float, float, float]:
  
  # Compute terms
  shot_theta = get_geometry(x_p, y_p)[0]
  if theta is None:
    goalie_theta = shot_theta
  else:
    goalie_theta = theta
  vertices = get_goalie_vertices(d_g, goalie_theta)
  apparent_area = get_net_apparent(shot_theta)
  proj = puck_persp_proj(vertices, x_p, y_p)  # Accounting for differences of the shot angle and goalie angle


  # Projected bounding intervals
  x_proj_min, x_proj_max = np.min(proj[:, 0]), np.max(proj[:, 0])
  z_proj_min, z_proj_max = np.min(proj[:, 1]), np.max(proj[:, 1]) 

  x_net_min, x_net_max = -W_NET / 2.0, W_NET / 2.0
  z_net_min, z_net_max = 0.0, H_NET

  # Compute exposed areas
  x_covered = max(0.0, min(x_proj_max, x_net_max) - max(x_proj_min, x_net_min))
  z_covered = max(0.0, min(z_proj_max, z_net_max) - max(z_proj_min, z_net_min))

  covered_area = x_covered * z_covered # Calculate the area of the net that is covered by the goalie
  exposed_area = max(0.0, apparent_area - covered_area) # Calculate the area of the net that is exposed to the shot
  if apparent_area > 1e-6:
    occlusion_ratio = covered_area / apparent_area
  else:
    occlusion_ratio = 1.0
  occlusion_ratio = float(np.clip(occlusion_ratio, 0.0, 1.0)) # Get the occlusion ratio, ensuring it is between 0 and 1

  return exposed_area, covered_area, occlusion_ratio

# ---------------------------------------
# STOCHASTIC STUFF - STATIC MODEL
# ---------------------------------------
def sample_release_neighbourhood(
    p0: Tuple[float, float], 
    sigma_x: float = SIGMA_X_DEFAULT, 
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES
) -> np.ndarray:

  # Sample a neighbourhood of release points around the original point p0
  sample_matrix = np.random.normal(loc=p0, scale=[sigma_x, sigma_y], size=(n_samples, 2))

  return sample_matrix

def expected_exposed_area_eff(
    p0: Tuple[float, float], 
    d_g: float, 
    samples: np.ndarray = None,
) -> Tuple[float, float, float]:
  
  # Evaluate expected exposed area and occlusion ratio over the neighbourhood of release points
  if samples is None:
    samples = sample_release_neighbourhood(p0)

  covered_areas = []  
  exposed_areas = []
  occlusion_ratios = []

  for sample in samples:
    x_p, y_p = sample
    exposed_area, covered_area, occlusion_ratio = exposed_area_eff(x_p, y_p, d_g)
    exposed_areas.append(exposed_area)
    covered_areas.append(covered_area)
    occlusion_ratios.append(occlusion_ratio)

  expected_exposed_area = np.mean(exposed_areas)
  expected_covered_area = np.mean(covered_areas)  
  expected_occlusion_ratio = np.mean(occlusion_ratios)

  return expected_exposed_area, expected_covered_area, expected_occlusion_ratio

def solve_dg_static(
    p0: Tuple[float, float],
    d_bounds: Tuple[float, float] = (0.0, R_CREASE),
    samples: np.ndarray = None
) -> float:
  
  # Solve for the optimal goalie distance d_g that minimizes the expected exposed area
  theta_set = get_geometry(p0[0], p0[1])[0]

  def obj(d_g: float) -> float:
    areas = []
    for sample in samples:
      x_p, y_p = sample
      exposed_area, _, _ = exposed_area_eff(x_p, y_p, d_g, theta_set)
      areas.append(exposed_area)
    return float(np.mean(areas))

  result = minimize_scalar(obj, bounds=d_bounds, method='bounded')

  return float(result.x)

# ---------------------------------------
# STOCHASTIC STUFF - DYNAMIC MODEL
# ---------------------------------------
def calc_pass_recovery(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    d_g: float,
) -> Tuple[float, float, float, float]:

  # Calculate the time it takes for the puck to travel from p1 to p2
  dist_pass = float(np.linalg.norm(np.array(p2) - np.array(p1)))
  t_pass = float(dist_pass / V_PASS)

  # Calculate the time it takes for the goalie to slide to the new position
  theta_1 = get_geometry(p1[0], p1[1])[0]
  theta_2 = get_geometry(p2[0], p2[1])[0]
  delta_theta = np.abs(theta_2 - theta_1)

  delta_s_g = 2.0 * d_g * np.sin(delta_theta / 2.0)  # Arc length for the goalie to slide
  t_slide = float(delta_s_g / V_SLIDE + T_REACT)

  # Calculate the time difference between the pass and the goalie's recovery
  delta_t = float(max(0.0, t_pass - t_slide))

  return t_pass, t_slide, delta_t, dist_pass

def calc_recovery_risk(
    p1: Tuple[float, float],
    p2_0: Tuple[float, float],
    d_g: float,
    samples: np.ndarray = None,
) -> float:
 
  # Calculate the probability of recovery risk
  if samples is None:
    samples = sample_release_neighbourhood(p2_0)

  if len(samples) == 0:
    return 0.0

  late_count = 0
  for pt in samples:
    _, _, delta_t, _ = calc_pass_recovery(p1, (pt[0], pt[1]), d_g)
    if delta_t > 0.0:
      late_count += 1

  risk = float(np.clip((late_count / len(samples)), 0.0, 1.0))

  return risk

def solve_dg_dynamic(
    p1: Tuple[float, float], 
    p2_0: Tuple[float, float], 
    samples: np.ndarray = None,
) -> float:
  
  # Solve for the optimal goalie distance d_g that minimizes the recovery risk
  if samples is None:
    samples = sample_release_neighbourhood(p2_0)

  if len(samples) == 0:
    return R_CREASE

  if calc_recovery_risk(p1, p2_0, 0.0, samples) > RISK_THRESHOLD:
    return 0.0

  lower_bound = 0.0
  upper_bound = R_CREASE

  if calc_recovery_risk(p1, p2_0, upper_bound, samples) <= RISK_THRESHOLD:
    return upper_bound

  for _ in range(40): # Bisecting to find the optimal d_g
    middle_bound = (lower_bound + upper_bound) / 2.0
    risk = calc_recovery_risk(p1, p2_0, middle_bound, samples)

    if risk <= RISK_THRESHOLD:
      lower_bound = middle_bound
    else:
      upper_bound = middle_bound

  return float(lower_bound)

# ---------------------------------------
# VISUALIZATION - STATIC MODEL
# ---------------------------------------
def plot_exposure_contour(
    x_grid: np.ndarray = None,
    y_grid: np.ndarray = None,
    d_g: float = None,
    p0: Tuple[float, float] = (0.0, 8.0),
    ax=None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):

  if x_grid is None:
    x_grid = np.linspace(-10.0, 10.0, 80)

  if y_grid is None:
    y_grid = np.linspace(0.0, 14.0, 80)

  if d_g is None:
    samples = sample_release_neighbourhood(
        p0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_samples=n_samples,
    )
    d_g = solve_dg_static(p0, samples=samples)

  theta_set = np.arctan2(p0[0], p0[1])

  X, Y = np.meshgrid(x_grid, y_grid)
  Z = np.zeros_like(X, dtype=float)

  for i in range(len(y_grid)):
    for j in range(len(x_grid)):
      exposed_area, _, _ = exposed_area_eff(
            float(X[i, j]), float(Y[i, j]), d_g, theta = theta_set
        )
      Z[i, j] = exposed_area

  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 5))
  else:
    fig = ax.figure

  contour = ax.contourf(X, Y, Z, levels=50, cmap='magma')
  fig.colorbar(contour, ax=ax, label='Exposed area $A_{eff}$ [m$^2$]')

  # The nominal release point is drawn to show the intended play spot beneath the
  # zone-wide surface.
  ax.scatter([p0[0]], [p0[1]], color='white', marker='x', s=60, linewidths=2, zorder=3)

  ax.set_title(f'Static exposure surface for $d_g={d_g:.2f}$ m')
  ax.set_xlabel('Offensive-zone x-position $x_p$ [m]')
  ax.set_ylabel('Offensive-zone y-position $y_p$ [m]')
  ax.grid(alpha=0.2)

  return fig, ax


def plot_release_cloud_exposure(
    p0: Tuple[float, float] = (0.0, 8.0),
    samples: np.ndarray = None,
    d_g: float = None,
    ax=None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Scatter the sampled release neighbourhood around p0 and colour each point by
  the perspective net exposure A_eff at that release coordinate.

  This is a direct visual of the stochastic modelling assumption: the shot is
  not a single point, but a cloud of plausible puck coordinates around the
  nominal play spot.

  Usage later in main.ipynb:
      fig, ax = model.plot_release_cloud_exposure(p0=(0.0, 8.0))
      plt.show()
  """

  if samples is None:
    samples = sample_release_neighbourhood(
        p0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_samples=n_samples,
    )

  if d_g is None:
    d_g = solve_dg_static(p0, samples=samples)

  exposed_values = []
  for sample in samples:
    x_p, y_p = sample
    exposed_area, _, _ = exposed_area_eff(x_p, y_p, d_g)
    exposed_values.append(exposed_area)

  if ax is None:
    fig, ax = plt.subplots(figsize=(7, 6))
  else:
    fig = ax.figure

  scatter = ax.scatter(
      samples[:, 0],
      samples[:, 1],
      c=exposed_values,
      cmap='plasma',
      s=12,
      alpha=0.75,
      edgecolors='none',
  )

  ax.scatter([p0[0]], [p0[1]], color='black', marker='x', s=60, linewidths=2, zorder=3)
  ax.set_xlim(p0[0] - 1.5, p0[0] + 1.5)
  ax.set_ylim(p0[1] - 1.5, p0[1] + 1.5)

  ax.set_title(
      f'Release cloud exposure for $d_g={d_g:.2f}$ m\n'
      f'Expected exposed area = {np.mean(exposed_values):.3f} m$^2$'
  )
  ax.set_xlabel('Release x-position $x_p$ [m]')
  ax.set_ylabel('Release y-position $y_p$ [m]')
  ax.grid(alpha=0.2)

  fig.colorbar(scatter, ax=ax, label='Exposed area $A_{eff}$ [m$^2$]')

  return fig, ax


def plot_goal_plane_heatmap(
    p0: Tuple[float, float] = (0.0, 8.0),
    d_g: float = None,
    theta_set: float = None,
    ax=None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Render the goal face from the shooter's perspective and overlay the projected
  goalie shadow.

  The red rectangle shows the physical goal frame W x H, while the blue polygon
  shows the part of that frame which is visually blocked by the goaltender at
  depth d_g. The effect is the direct geometric interpretation of the model's
  uncovered vs. covered net aperture.

  Usage later in main.ipynb:
      fig, ax = model.plot_goal_plane_heatmap(p0=(0.0, 8.0), d_g=0.6)
      plt.show()
  """

  if theta_set is None:
    theta_set, _ = get_geometry(p0[0], p0[1])

  if d_g is None:
    samples = sample_release_neighbourhood(
        p0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_samples=n_samples,
    )
    d_g = solve_dg_static(p0, samples=samples)

  vertices = get_goalie_vertices(d_g, theta_set)
  projected = puck_persp_proj(vertices, p0[0], p0[1])

  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
  else:
    fig = ax.figure

  goal_frame = plt.Rectangle(
      (-W_NET / 2.0, 0.0),
      W_NET,
      H_NET,
      linewidth=2,
      edgecolor='red',
      facecolor='none',
      label='Goal frame',
  )
  ax.add_patch(goal_frame)

  goalie_shadow = Polygon(
      projected,
      closed=True,
      facecolor='tab:blue',
      edgecolor='black',
      alpha=0.55,
      label='Projected goalie shadow',
  )
  ax.add_patch(goalie_shadow)

  ax.set_title(f'Goal-plane view for $d_g={d_g:.2f}$ m')
  ax.set_xlabel('Goal width $x$ [m]')
  ax.set_ylabel('Goal height $z$ [m]')
  ax.set_xlim(-W_NET / 2.0, W_NET / 2.0)
  ax.set_ylim(0.0, H_NET)
  ax.set_aspect('equal')
  ax.grid(alpha=0.2)

  return fig, ax


# ---------------------------------------
# VISUALIZATION - DYNAMIC MODEL
# ---------------------------------------
def plot_recovery_window(
    p1: Tuple[float, float],
    p2_0: Tuple[float, float],
    d_grid: np.ndarray = None,
    ax=None,
):
  """
  Plot the pass duration and goalie recovery duration against crease depth.

  This is the dynamic visual that clarifies the actual meaning of the recovery
  model: the puck travel time is fixed by the pass geometry, while the goalie
  recovery time grows linearly with the distance the goalie must slide. The
  shaded region therefore shows where the goaltender cannot beat the pass.

  Usage later in main.ipynb:
      fig, ax = model.plot_recovery_window(p1=(-5.0, 7.0), p2_0=(5.0, 7.0))
      plt.show()
  """

  if d_grid is None:
    d_grid = np.linspace(0.0, R_CREASE, 80)

  theta_1, _ = get_geometry(p1[0], p1[1])
  theta_2, _ = get_geometry(p2_0[0], p2_0[1])

  dist_pass = float(np.linalg.norm(np.array(p2_0) - np.array(p1)))
  t_pass = dist_pass / V_PASS

  # Goalie recovery grows with the arc-length of the slide along the crease.
  t_goalie = T_REACT + (2.0 * d_grid * np.sin(np.abs(theta_2 - theta_1) / 2.0)) / V_SLIDE
  delta_t = np.maximum(0.0, t_goalie - t_pass)

  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 5))
  else:
    fig = ax.figure

  ax.axhline(t_pass, color='tab:blue', linewidth=2, label='Pass transit time $t_{pass}$')
  ax.plot(d_grid, t_goalie, color='tab:orange', linewidth=2, label='Recovery time $t_{goalie}(d_g)$')

  ax.fill_between(
      d_grid,
      t_pass,
      t_goalie,
      where=t_goalie > t_pass,
      color='tab:red',
      alpha=0.25,
      label='Late recovery region ($\delta t > 0$)',
  )

  ax.set_title(f'Recovery window for pass from $p_1$ to $p_2$')
  ax.set_xlabel('Goaltender depth $d_g$ [m]')
  ax.set_ylabel('Time [s]')
  ax.grid(alpha=0.2)
  ax.legend()

  return fig, ax


def plot_recovery_risk_curve(
    p1: Tuple[float, float],
    p2_0: Tuple[float, float],
    samples: np.ndarray = None,
    d_grid: np.ndarray = None,
    risk_threshold: float = RISK_THRESHOLD,
    ax=None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Plot the recovery-risk curve P_late(d_g) over the useful crease-depth range.

  The curve shows the probability that the receiving shooter is exposed because
  the goaltender cannot recover in time after the pass. The horizontal line is the
  acceptable risk threshold, and the stopping depth d_g^* is the largest depth
  where the risk remains within that tolerance.

  Usage later in main.ipynb:
      fig, ax = model.plot_recovery_risk_curve(p1=(-5.0, 7.0), p2_0=(5.0, 7.0))
      plt.show()
  """

  if samples is None:
    samples = sample_release_neighbourhood(
        p2_0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_samples=n_samples,
    )

  if d_grid is None:
    d_grid = np.linspace(0.0, R_CREASE, 80)

  risks = [calc_recovery_risk(p1, p2_0, d_g, samples=samples) for d_g in d_grid]
  d_star = solve_dg_dynamic(p1, p2_0, samples=samples)

  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 5))
  else:
    fig = ax.figure

  ax.plot(d_grid, risks, color='tab:purple', linewidth=2, label='Late recovery probability $P_{late}(d_g)$')
  ax.axhline(risk_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Risk threshold = {risk_threshold:.2f}')

  ax.scatter(
      [d_star],
      [calc_recovery_risk(p1, p2_0, d_star, samples=samples)],
      color='white',
      edgecolors='black',
      s=80,
      zorder=3,
      label=f'$d_g^*={d_star:.2f}$ m',
  )

  ax.set_title('Recovery risk curve across the crease depth range')
  ax.set_xlabel('Goaltender depth $d_g$ [m]')
  ax.set_ylabel('Recovery risk $P_{late}$')
  ax.grid(alpha=0.2)
  ax.legend()

  return fig, ax


def plot_recovery_risk_contour(
    p1: Tuple[float, float],
    p2_0: Tuple[float, float],
    d_grid: np.ndarray = None,
    x_grid: np.ndarray = None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Legacy contour view of recovery risk over a 2D grid of receiving locations.

  This is a broader survey visual than the 1D risk curve. It is useful when
  you want to see how the late-recovery probability changes as the receiving
  puck moves laterally and the goalie depth is adjusted.
  """

  if x_grid is None:
    x_grid = np.linspace(p2_0[0] - 2.0, p2_0[0] + 2.0, 35)

  if d_grid is None:
    d_grid = np.linspace(0.0, R_CREASE, 35)

  risk_surface = np.zeros((len(d_grid), len(x_grid)))

  for i, x_pos in enumerate(x_grid):
    samples = sample_release_neighbourhood(
        (x_pos, p2_0[1]),
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        n_samples=n_samples,
    )

    for j, d_g in enumerate(d_grid):
      risk_surface[j, i] = calc_recovery_risk(p1, (x_pos, p2_0[1]), d_g, samples=samples)

  fig, ax = plt.subplots(figsize=(8, 5))
  contour = ax.contourf(x_grid, d_grid, risk_surface, levels=50, cmap='magma')
  fig.colorbar(contour, ax=ax, label='Recovery risk $P_{late}$')

  ax.contour(
      x_grid,
      d_grid,
      risk_surface,
      levels=[RISK_THRESHOLD],
      colors='white',
      linewidths=1.5,
      linestyles='--',
  )

  ax.set_title('Dynamic recovery-risk contour')
  ax.set_xlabel('Receiving puck $x$-position [m]')
  ax.set_ylabel('Goaltender depth $d_g$ [m]')
  ax.grid(alpha=0.2)

  return fig, ax


def plot_zone_depth_surface(
    p1: Tuple[float, float],
    grid_x: np.ndarray = None,
    grid_y: np.ndarray = None,
    ax=None,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Solve for the optimal depth d_g^* on a 2D mesh of candidate pass-receive
  coordinates, and render the resulting top-down depth surface.

  This is the zone-wide map that answers the coaching question 'where can the
  goalie challenge aggressively, and where do they need to stay close to the
  goal line?'

  Usage later in main.ipynb:
      fig, ax = model.plot_zone_depth_surface(p1=(-5.0, 7.0))
      plt.show()
  """

  if grid_x is None:
    grid_x = np.linspace(-6.0, 6.0, 40)

  if grid_y is None:
    grid_y = np.linspace(0.0, 14.0, 40)

  X, Y = np.meshgrid(grid_x, grid_y)
  surface = np.zeros_like(X, dtype=float)

  for i in range(len(grid_y)):
    for j in range(len(grid_x)):
      samples = sample_release_neighbourhood(
          (float(X[i, j]), float(Y[i, j])),
          sigma_x=sigma_x,
          sigma_y=sigma_y,
          n_samples=n_samples,
      )
      surface[i, j] = solve_dg_dynamic(p1, (float(X[i, j]), float(Y[i, j])), samples=samples)

  if ax is None:
    fig, ax = plt.subplots(figsize=(7, 6))
  else:
    fig = ax.figure

  image = ax.imshow(
      surface,
      extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
      origin='lower',
      cmap='inferno',
      aspect='auto',
  )

  ax.set_title('Zone-wide optimal crease depth map')
  ax.set_xlabel('Receiving puck $x$-position [m]')
  ax.set_ylabel('Receiving puck $y$-position [m]')
  fig.colorbar(image, ax=ax, label='Optimal depth $d_g^*$ [m]')

  return fig, ax


def plot_zone_depth_heatmap(
    p1: Tuple[float, float],
    p2_0: Tuple[float, float],
    x_extent: float = 4.0,
    y_extent: float = 4.0,
    resolution: int = 24,
    sigma_x: float = SIGMA_X_DEFAULT,
    sigma_y: float = SIGMA_Y_DEFAULT,
    n_samples: int = N_SAMPLES,
):
  """
  Backward-compatible wrapper around the zone-depth surface renderer.

  This keeps the earlier API intact while making the underlying surface
  interpretation more explicit.
  """

  x_grid = np.linspace(p2_0[0] - x_extent, p2_0[0] + x_extent, resolution)
  y_grid = np.linspace(p2_0[1] - y_extent, p2_0[1] + y_extent, resolution)

  return plot_zone_depth_surface(
      p1,
      grid_x=x_grid,
      grid_y=y_grid,
      sigma_x=sigma_x,
      sigma_y=sigma_y,
      n_samples=n_samples,
  )
