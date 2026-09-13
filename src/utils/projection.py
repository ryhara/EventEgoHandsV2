import os
if os.name != 'nt': os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import torch

from .config import OUTPUT_HEIGHT, OUTPUT_WIDTH

MAIN_CAMERA = pyrender.PerspectiveCamera(yfov=np.deg2rad(30), aspectRatio=OUTPUT_WIDTH / OUTPUT_HEIGHT)
PROJECTION_MATRIX = MAIN_CAMERA.get_projection_matrix(OUTPUT_WIDTH, OUTPUT_HEIGHT)


def opengl_projection_transform(opengl_projection_matrix, width, height, points):
    if isinstance(points, np.ndarray):
        points = np.concatenate([points, np.ones((points.shape[0], 1))], -1)
    elif isinstance(points, torch.Tensor):
        shape = points.shape[:-1]
        device = points.device

        points = torch.cat([points, torch.ones(*shape, 1).to(device)], -1)

    shape = points.shape[:-1]

    if isinstance(points, np.ndarray):
        points = np.reshape(points, (-1, 4))
    else:
        points = points.view(-1, 4)

    h_points = (opengl_projection_matrix @ points.T).T
    h_points = h_points / h_points[:, -1:]

    h_points = (1 - h_points) * 0.5
    h_points[:, 0] *= width
    h_points[:, 1] *= height

    if isinstance(h_points, np.ndarray):
        h_points = np.reshape(h_points, (*shape, 4))
    else:
        h_points = h_points.view(*shape, 4)

    return h_points[..., :2]
