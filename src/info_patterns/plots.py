from collections.abc import Sequence
from typing import Any

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap

def set_plot_style(usetex: bool, font_family: str, figsize: tuple[float, float]) -> None:
    mpl.rc("text", usetex=usetex)
    mpl.rc("font", family=font_family)
    mpl.rc("figure", figsize=figsize)

def set_axes_equal(ax: Axes, scale: float) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = scale * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    for ctr, dim in zip(origin, ['x', 'y', 'z']):
        getattr(ax, f'set_{dim}lim')(ctr - radius, ctr + radius)

def add_mini_axis(fig: Figure, position: Sequence[float], axis_len: float, elev: float, azim: float, fontsize: float) -> Axes:
    ax_orient = fig.add_axes(position, projection='3d')
    ax_orient.quiver(0, 0, 0, axis_len, 0, 0, color="black", arrow_length_ratio=0.18, linewidth=1)
    ax_orient.quiver(0, 0, 0, 0, -axis_len, 0, color="black", arrow_length_ratio=0.18, linewidth=1)
    ax_orient.quiver(0, 0, 0, 0, 0, axis_len, color="black", arrow_length_ratio=0.18, linewidth=1)

    ax_orient.text(axis_len + 0.1, -0.12, 0, r"$x$", fontsize=fontsize)
    ax_orient.text(0, -axis_len - 0.4, 0, r"$y$", fontsize=fontsize)
    ax_orient.text(-0.1, 0, axis_len + 0.08, r"$z$", fontsize=fontsize)

    ax_orient.set_xlim(0, 1.2)
    ax_orient.set_ylim(-1.2, 0)
    ax_orient.set_zlim(0, 1.2)

    ax_orient.set_xticks([])
    ax_orient.set_yticks([])
    ax_orient.set_zticks([])

    ax_orient.grid(False)

    ax_orient.xaxis.pane.fill = False
    ax_orient.yaxis.pane.fill = False
    ax_orient.zaxis.pane.fill = False

    ax_orient.xaxis.pane.set_edgecolor("white")
    ax_orient.yaxis.pane.set_edgecolor("white")
    ax_orient.zaxis.pane.set_edgecolor("white")

    ax_orient.xaxis.line.set_color((1, 1, 1, 0))
    ax_orient.yaxis.line.set_color((1, 1, 1, 0))
    ax_orient.zaxis.line.set_color((1, 1, 1, 0))

    ax_orient.view_init(elev=elev, azim=azim)

    return ax_orient

def plot3d_info_patterns(Nteta: int, Nphi: int, info_patterns: Sequence[np.ndarray], labels: Sequence[str], cmap: Colormap, scale_list: Sequence[float] | None, savefig: bool, 
                         save_type: str, filename: str, results_dir: str) -> None:
    theta = np.linspace(0, np.pi, Nteta)
    phi = np.linspace(0, 2 * np.pi, Nphi)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    global_max = np.max([np.max(I) for I in info_patterns])

    fig = plt.figure(figsize=(8, 3.5), dpi=300)
    axes = []

    norm = plt.Normalize(vmin=0, vmax=1)

    if scale_list is None:
        scale_list = [0.47] * len(info_patterns)

    for i, (I, label, scale_value) in enumerate(zip(info_patterns, labels, scale_list)):
        r = I / np.max(I)
        I_color = I / global_max

        X = r * np.sin(Theta) * np.cos(Phi)
        Y = r * np.sin(Theta) * np.sin(Phi)
        Z = r * np.cos(Theta)

        ax = fig.add_subplot(1, len(info_patterns), i + 1, projection='3d')
        axes.append(ax)

        ax.grid(False)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

        ax.set_box_aspect((1, 1, 1))

        facecolors = cmap(norm(I_color))
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, linewidth=0.2, antialiased=True)
        ax.set_title(fr'$I_{{{label}}}(\theta,\phi)$', fontsize=14)
        set_axes_equal(ax, scale=scale_value)

    fig.subplots_adjust(left=0.02, right=0.86, top=0.90, bottom=0.08, wspace=0.01)

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cax = fig.add_axes([0.89, 0.47, 0.015, 0.30])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Intensity")

    add_mini_axis(fig, position=[0.855, 0.25, 0.12, 0.20], axis_len=0.65, elev=20, azim=-60, fontsize=11)

    if savefig:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{filename}.{save_type}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    plt.show()

def plot2d_info_patterns(Nteta: int, Nphi: int, info_patterns: Sequence[np.ndarray], labels: Sequence[str], planes: Sequence[str], normalize: bool, fill: bool, 
                         figsize: tuple[float, float] | None, savefig: bool, save_type: str, filename: str, results_dir: str) -> None:
    theta_grid = np.linspace(0, np.pi, Nteta)
    phi_grid = np.linspace(0, 2 * np.pi, Nphi)
    alpha = np.linspace(0, 2 * np.pi, 720)
    plane_colors = {"xy": "black", "xz": "tab:red", "yz": "tab:blue"}

    if figsize is None:
        figsize = (3.2 * len(info_patterns), 3.2)

    fig, axes = plt.subplots(1, len(info_patterns), figsize=figsize, dpi=300, subplot_kw={"projection": "polar"})

    if len(info_patterns) == 1:
        axes = [axes]

    def nearest_index(array: np.ndarray, value: float) -> int:
        return np.argmin(np.abs(array - value))

    def sample_pattern(I: np.ndarray, plane: str) -> np.ndarray:
        r_values = []

        for a in alpha:
            if plane == "xy":
                theta = np.pi / 2
                phi = a

            elif plane == "xz":
                x = np.cos(a)
                z = np.sin(a)

                theta = np.arccos(z)
                phi = 0 if x >= 0 else np.pi

            elif plane == "yz":
                y = np.cos(a)
                z = np.sin(a)

                theta = np.arccos(z)
                phi = np.pi / 2 if y >= 0 else 3 * np.pi / 2

            else:
                raise ValueError("planes must contain only 'xy', 'xz', or 'yz'.")

            itheta = nearest_index(theta_grid, theta)
            iphi = nearest_index(phi_grid, phi)

            r_values.append(I[itheta, iphi])

        return np.array(r_values)

    for ax, I, label in zip(axes, info_patterns, labels):

        if normalize:
            I_plot = I / np.max(I)
        else:
            I_plot = I.copy()

        for plane in planes:
            r = sample_pattern(I_plot, plane)
            ax.plot(alpha, r, lw=1.5, color=plane_colors[plane])
            if fill:
                ax.fill(alpha, r, color=plane_colors[plane], alpha=0.18)

        ax.set_title(fr"$I_{{{label}}}$", fontsize=14)
        ax.set_ylim(0, 1.05 if normalize else None)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.spines["polar"].set_linewidth(0.8)

    legend_handles = [plt.Line2D([0], [0], color=plane_colors[plane], lw=1.5, label=plane) for plane in planes]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    if savefig:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{filename}.{save_type}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    plt.show()


def plot_force_displacement(displacements: np.ndarray, Force: dict[str, np.ndarray], savefig: bool, save_type: str, filename: str, results_dir: str) -> tuple[Figure, np.ndarray]:
    components = ["X", "Y", "Z"]
    markers = ["x", "o", "s"]
    colors = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)
    for j, axis in enumerate(components):
        for i, comp in enumerate(components):
            axs[j].plot(displacements, Force[axis][:, i], marker=markers[i], color=colors[i], label=fr"$F_{comp}$")
        axs[j].set_title(fr"Displacement along ${axis.lower()}$")
        axs[j].set_xlabel(r"Displacement [nm]")
        axs[j].set_ylabel(r"Force [N]")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if savefig:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{filename}.{save_type}")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    plt.show()

    return fig, axs

def plot_torque_rotation(angles_deg: np.ndarray, Torque: dict[str, np.ndarray], savefig: bool, save_type: str, filename: str, results_dir: str) -> tuple[Figure, np.ndarray]:
    components = ["X", "Y", "Z"]
    markers = ["x", "o", "s"]
    colors = [cm.viridis(x) for x in [0.1, 0.5, 0.9]]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)
    for j, axis in enumerate(components):
        for i, comp in enumerate(components):
            axs[j].plot(angles_deg, Torque[axis][:, i], marker=markers[i], color=colors[i], label=fr"$\tau_{comp}$")
        axs[j].set_title(fr"Rotation around ${axis.lower()}$")
        axs[j].set_xlabel(r"Angle [deg]")
        axs[j].set_ylabel(r"Torque [N m]")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if savefig:
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"{filename}.{save_type}")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {save_path}")

    plt.show()

    return fig, axs