# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from pymol import cmd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import plotly.graph_objects as go
import torch
import os
from pathlib import Path
from pymol.cgo import BEGIN, COLOR, TRIANGLES, VERTEX, NORMAL, END

import math
import numpy as np

def vertex(a1, a2, a3, u, v, M, r0):
    #https://github.com/vlasenkov/pymol-iellipsoid/blob/master/iellipsoid.py
    vrtx = M.dot(np.array([
        a1 * math.cos(u) * math.cos(v),
        a2 * math.cos(u) * math.sin(v),
        a3 * math.sin(u)
        ]))
    nrml = M.dot(np.array([
        math.cos(u) * math.cos(v) / a1,
        math.cos(u) * math.sin(v) / a2,
        math.sin(u) / a3
        ]))
    return vrtx + r0, nrml


def draw_ellipse(r0, C, color=[0.5, 0.5, 0.5], transparency=0.0, name='', resolution=12):
    #https://github.com/vlasenkov/pymol-iellipsoid/blob/master/iellipsoid.py
    ws, vs = np.linalg.eig(C)
    # M = np.linalg.inv(vs)
    M = vs
    a1, a2, a3 = np.sqrt(ws)
    u_segs = resolution
    v_segs = resolution
    mesh = [BEGIN, TRIANGLES, COLOR]
    mesh.extend(color)
    dU = math.pi / u_segs
    dV = 2 * math.pi / v_segs
    U = -math.pi / 2
    for Y in range(0, u_segs):
        V = math.pi
        for X in range(0, v_segs):
    
            (x1, y1, z1), (n1x, n1y, n1z) = vertex(a1, a2, a3,
                                                   U, V, M, r0)
            (x2, y2, z2), (n2x, n2y, n2z) = vertex(a1, a2, a3,
                                                   U + dU, V, M, r0)
            (x3, y3, z3), (n3x, n3y, n3z) = vertex(a1, a2, a3,
                                                   U + dU, V + dV,
                                                   M, r0)
            (x4, y4, z4), (n4x, n4y, n4z) = vertex(a1, a2, a3,
                                                   U, V + dV, M, r0)
    
            mesh.extend([NORMAL, n1x, n1y, n1z, VERTEX, x1, y1, z1])
            mesh.extend([NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
            mesh.extend([NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
            mesh.extend([NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
            mesh.extend([NORMAL, n3x, n3y, n3z, VERTEX, x3, y3, z3])
            mesh.extend([NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
    
            V += dV
        U += dU
    mesh.append(END)
    cmd.load_cgo(mesh, name)
    cmd.set("cgo_transparency", transparency, name)


def visualize_blobs(tmp_paths, means, covars, save_path=None, ax=None):
    cmd.reinitialize()
    for mean, covar in zip(means, covars):
        draw_ellipse(mean, 5 * covar, transparency=0.5)
    for tmp_path in tmp_paths:    
        cmd.load(tmp_path)
        os.remove(tmp_path)
    cmd.orient(Path(tmp_paths[-1]).stem)
    #cmd.spectrum()
    cmd.bg_color('white')
    cmd.set('ray_trace_mode', 0)
    cmd.set('depth_cue', 'off')
    cmd.set('ray_shadows', 'off')
    
    if save_path is not None:
        cmd.save(save_path)
    if ax is not None:
        cmd.png('/tmp/tmp.png', 640, 640)
        img = plt.imread('/tmp/tmp.png')
        os.remove('/tmp/tmp.png')
            
        if ax is None:
            plt.imshow(img)
            plt.axis('off')  # Optional: Hide axis ticks and labels
            plt.show()
        else:
            ax.imshow(img)
            ax.axis('off')

def plot_point_cloud(point_clouds, path=None):
    # Takes a list of point cloud tensors and plots them
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]

    colors = ['red', 'blue', 'green', 'yellow', 'orange']  # List of colors for each point cloud
    traces = []  # List to store individual traces for each point cloud

    for i, point_cloud in enumerate(point_clouds):
        if isinstance(point_cloud, np.ndarray):
            pass
        elif isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.numpy()

        x_data = point_cloud[:, 0]
        y_data = point_cloud[:, 1]
        z_data = point_cloud[:, 2]

        # Create a trace for each point cloud with a different color
        trace = go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
                color=colors[i % len(colors)]  # Assign color based on the index of the point cloud
            ),
            name=f"Point Cloud {i + 1}"
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        )
    )

    # Create the figure and add the traces to it
    fig = go.Figure(data=traces, layout=layout)

    if path is None:
        fig.show()
    else:
        fig.write_html(path)