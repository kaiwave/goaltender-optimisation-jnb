from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# CONSTANTS, CONSTRAINTS & PARAMETERS
# Define constraints and parameters which will be used throughout the model.
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

# Stochastic Spatial Dispersion (Standard Deviations in meters)
SIGMA_X_DEFAULT: float = 0.40  # Lateral stickhandling release variance
SIGMA_Y_DEFAULT: float = 0.30  # Longitudinal stride release variance

# CORE ENGINE - GEOMETRY AND PROJECTION
def get_geometry(x_p: float, y_p: float) -> tuple[float, float]:
  # Calc theta and distance from the point to the net
  theta = np.arctan2(x_p, y_p)
  distance_Ds = np.sqrt(x_p**2 + y_p**2) 

  return theta, distance_Ds

def get_net_apparent(theta: float) -> float:
  # Calc apparent area of the net from the angle theta
  apparent_area = W_NET * H_NET * np.cos(theta)

  return apparent_area

def get_goalie_vertices(d_g: float, theta: float) -> np.ndarray:
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

def puck_persp_proj(goalie_vertices: np.ndarray, x_p: float, y_p: float) -> np.ndarray:
  # Project the goalie vertices onto the goal plane

  x_v = goalie_vertices[:, 0]
  y_v = goalie_vertices[:, 1]
  z_v = goalie_vertices[:, 2] # Take goalie vertices since they are already sorted

  den = np.where(np.abs(y_p - y_v) < 1e-6, 1e-6, y_p - y_v) # Check for divide by zero
  t = y_p / den

  x_proj = x_p - t * (x_p - x_v)
  z_proj = t * z_v # Arithmetic from the projection formula in 2.2.4

  return np.column_stack((x_proj, z_proj)) # Returns a 2D array of size (4, 2) with the projected coordinates

def exposed_area_eff(x_p: float, y_p: float, d_g: float, theta: float = None) -> Dict[str, float]:
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

  covered_area = x_covered * z_covered
  exposed_area = max(0.0, apparent_area - covered_area)
  if apparent_area > 1e-6:
    occlusion_ratio = covered_area / apparent_area
  else:
    occlusion_ratio = 1.0
  occlusion_ratio = float(np.clip(occlusion_ratio, 0.0, 1.0))

  exposed_metrics = {
    "exposed_area": float(exposed_area),
    "covered_area": float(covered_area),
    "occlusion_ratio": occlusion_ratio,
  }
  return exposed_metrics