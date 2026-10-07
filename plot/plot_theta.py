"""
Plot the R^2 -> R test function:

    theta(xi) = 0.5*(xi1^2 + xi2^2)
                + 100*(sin^2(pi*xi1/2) + sin^2(pi*xi2/2))

Produces a 3D surface plot and a 2D contour/heatmap plot side by side.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)
import seaborn as sns

sns.set_theme(style="whitegrid")


def theta(xi1, xi2):
    return 0.5 * (xi1**2 + xi2**2) + 100.0 * (
        np.sin(np.pi * xi1 / 2.0) ** 2 + np.sin(np.pi * xi2 / 2.0) ** 2
    )


# Grid of evaluation points
lo, hi, n = -11.0, 11.0, 400
x1 = np.linspace(lo, hi, n)
x2 = np.linspace(lo, hi, n)
X1, X2 = np.meshgrid(x1, x2)
Z = theta(X1, X2)

def make_surface_fig(colorbar):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0, antialiased=True)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    if colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    fig.tight_layout()
    return fig


def make_contour_fig(colorbar):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    contour = ax.contourf(X1, X2, Z, levels=60, cmap="viridis")
    ax.contour(X1, X2, Z, levels=15, colors="k", linewidths=0.3, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(contour, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig


make_surface_fig(colorbar=True).savefig("theta_surface_cbar.png", dpi=150)
make_surface_fig(colorbar=False).savefig("theta_surface.png", dpi=150)
make_contour_fig(colorbar=True).savefig("theta_contour_cbar.png", dpi=150)
make_contour_fig(colorbar=False).savefig("theta_contour.png", dpi=150)
plt.show()
