
#!/usr/bin/env python3
"""Plot the 43 Sierra / near-Sierra SNOTEL station locations with a U.S. inset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

import cartopy.crs as ccrs
REPO_ROOT = Path("/global/homes/h/hyvchen/Snow-Predication-at-Sierra")
SIERRA_ROOT = REPO_ROOT / "experiments" / "foreswe_sierra_station"
DATA_PATH = SIERRA_ROOT / "data_subset" / "snotel_location_info_2.csv"
FIGURE_DIR = SIERRA_ROOT / "figures"
PNG_PATH = FIGURE_DIR / "sierra_station_locations_map.png"
PDF_PATH = FIGURE_DIR / "sierra_station_locations_map.pdf"

BOX_LAT_MIN = 34.5
BOX_LAT_MAX = 42.5
BOX_LON_MIN = -124.0
BOX_LON_MAX = -116.0

MAIN_EXTENT = [-124.6, -116.0, 35.0, 42.8]
INSET_EXTENT = [-126.0, -100.0, 30.0, 50.0]


def draw_state_lines(ax: plt.Axes, linewidth: float) -> None:
    proj = ccrs.PlateCarree()

    # California / Oregon
    ax.plot(
        [-124.35, -120.0],
        [42.0, 42.0],
        color="black",
        linewidth=linewidth,
        transform=proj,
        zorder=6,
    )

    # California / Nevada
    ax.plot(
        [-120.0, -120.0],
        [42.0, 39.0],
        color="black",
        linewidth=linewidth,
        transform=proj,
        zorder=6,
    )
    ax.plot(
        [-120.0, -114.6],
        [39.0, 35.0],
        color="black",
        linewidth=linewidth,
        transform=proj,
        zorder=6,
    )

    # Nevada / Oregon
    ax.plot(
        [-117.0, -117.0],
        [42.0, 42.02],
        color="black",
        linewidth=linewidth,
        transform=proj,
        zorder=6,
    )

    # Oregon / Nevada short visible top corner hint
    ax.plot(
        [-120.0, -117.0],
        [42.0, 42.0],
        color="black",
        linewidth=linewidth,
        transform=proj,
        zorder=6,
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    counts = df["State.Code"].value_counts().sort_index()

    fig = plt.figure(figsize=(14, 10))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)

    ax.stock_img()
    ax.set_extent(MAIN_EXTENT, crs=proj)
    draw_state_lines(ax, linewidth=2.2)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.35, color="#9A9A9A", alpha=0.35, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    ax.scatter(
        df["Longitude"],
        df["Latitude"],
        s=180,
        color="#D7263D",
        edgecolor="#7A0C16",
        linewidth=1.2,
        alpha=0.92,
        transform=proj,
        zorder=7,
        label="Sierra subset stations",
    )

    ax.set_title("Sierra / Near-Sierra SNOTEL Stations (43 Stations)", fontsize=15, pad=10)

    legend = ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#BBBBBB")
    for text in legend.get_texts():
        text.set_fontsize(10)

    summary = ", ".join(f"{state}={count}" for state, count in counts.items())
    ax.text(
        0.015,
        0.02,
        f"State counts: {summary}",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "boxstyle": "round,pad=0.3", "alpha": 0.95},
        zorder=10,
    )

    inset = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=0.7, axes_class=type(ax), axes_kwargs={"projection": proj})
    inset.stock_img()
    inset.set_extent(INSET_EXTENT, crs=proj)
    draw_state_lines(inset, linewidth=1.1)
    inset.scatter(
        df["Longitude"],
        df["Latitude"],
        s=42,
        color="#D7263D",
        edgecolor="#7A0C16",
        linewidth=0.45,
        alpha=0.9,
        transform=proj,
        zorder=7,
    )
    inset.plot(
        [MAIN_EXTENT[0], MAIN_EXTENT[1], MAIN_EXTENT[1], MAIN_EXTENT[0], MAIN_EXTENT[0]],
        [MAIN_EXTENT[2], MAIN_EXTENT[2], MAIN_EXTENT[3], MAIN_EXTENT[3], MAIN_EXTENT[2]],
        color="black",
        linewidth=1.5,
        transform=proj,
        zorder=8,
    )
    inset.set_title("Western U.S. context", fontsize=10, pad=3)

    con1 = ConnectionPatch(
        xyA=(MAIN_EXTENT[1], MAIN_EXTENT[3]),
        coordsA=ax.transData,
        xyB=(MAIN_EXTENT[1], MAIN_EXTENT[3]),
        coordsB=inset.transData,
        color="black",
        linewidth=1.2,
    )
    con2 = ConnectionPatch(
        xyA=(MAIN_EXTENT[0], MAIN_EXTENT[2]),
        coordsA=ax.transData,
        xyB=(MAIN_EXTENT[0], MAIN_EXTENT[2]),
        coordsB=inset.transData,
        color="black",
        linewidth=1.2,
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print("Saved PNG:", PNG_PATH)
    print("Saved PDF:", PDF_PATH)
    print("State counts:", counts.to_dict())


if __name__ == "__main__":
    main()

