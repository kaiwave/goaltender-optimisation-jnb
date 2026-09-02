import numpy as np
import matplotlib.pyplot as plt

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

# STATIC GEOMETRY AND PROJECTION
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
  x1 = (d_g * np.sin(theta)) - ((W_GOALIE / 2) * np.cos(theta))
  y1 = (d_g * np.cos(theta)) - ((W_GOALIE / 2) * -1 * np.sin(theta))
  v1 = np.array([x1, y1, 0])

  x2 = (d_g * np.sin(theta)) + ((W_GOALIE / 2) * np.cos(theta))
  y2 = (d_g * np.cos(theta)) + ((W_GOALIE / 2) * -1 * np.sin(theta))
  v2 = np.array([x2, y2, 0])

  v3 = np.array([x1, y1, H_GOALIE])
  v4 = np.array([x2, y2, H_GOALIE])

  goalie_vertices = np.array([v1, v2, v3, v4])
  return goalie_vertices