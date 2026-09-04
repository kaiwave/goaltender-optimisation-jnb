import numpy as np
import matplotlib.pyplot as plt
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

  for _ in range(50): # Bisecting to find the optimal d_g
    middle_bound = (lower_bound + upper_bound) / 2.0
    risk = calc_recovery_risk(p1, p2_0, middle_bound, samples)

    if risk <= RISK_THRESHOLD:
      lower_bound = middle_bound
    else:
      upper_bound = middle_bound

  return float(lower_bound)

# ---------------------------------------
# VISUALIZATION
# ---------------------------------------