"""Tests for file reading, zone contribution and plotting helpers."""
import numpy as np
import geopandas as gpd
from shapely.geometry import box

import configure11_functions as cf


def test_read_text_file(tmp_path):
    path = tmp_path / "GreenAreas.txt"
    path.write_text("2\nshape a\nshape b\n")
    assert cf.read_text_file(str(path)) == "2\nshape a\nshape b\n"


def test_read_interventions_counts_files_and_ignores_desktop_ini(tmp_path):
    for i in range(3):
        (tmp_path / f"zone_{i}.txt").write_text(f"1\nfeature {i}")
    (tmp_path / "desktop.ini").write_text("[.ShellClassInfo]")
    count, contents = cf.read_interventions(str(tmp_path))
    assert count == 3
    assert sorted(contents) == ["1\nfeature 0", "1\nfeature 1", "1\nfeature 2"]


def _zones(n):
    return gpd.GeoDataFrame({"cluster_id": range(1, n + 1)},
                            geometry=[box(i, 0, i + 1, 1) for i in range(n)])


def test_zone_contribution_percentages():
    population = np.array([[1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0],
                           [1, 1, 0, 0]])
    joined, column = cf.zone_contribution(_zones(4), "cluster_id", 4, population, 4)
    assert column == "per_count"
    assert list(joined["zone_id"]) == [1, 2, 3, 4]
    assert list(joined["count"]) == [4, 2, 1, 0]
    assert list(joined["per_count"]) == [100, 50, 25, 0]


def test_scatter_plot_map_plot_writes_png(tmp_path):
    population = np.random.RandomState(0).randint(2, size=(10, 6))
    zones = gpd.GeoDataFrame(
        {"cluster_id": range(1, 7)},
        geometry=[box(421000 + 500 * i, 563000, 421500 + 500 * i, 566000) for i in range(6)])
    joined, column = cf.zone_contribution(zones, "cluster_id", 6, population, 10)
    out = tmp_path / "Generation_No_0"
    cf.scatter_plot_map_plot("default", "Generation 0", np.arange(5.0), np.arange(5.0),
                             "Evolving Solution", [-1, 6], [-1, 6], "Cost", "EAD",
                             "Zones Contribution", "cividis_r",
                             np.array([10, 30, 50, 70, 90]), "Contribution (%)",
                             joined, column, str(out))
    png = tmp_path / "Generation_No_0.png"
    assert png.exists() and png.stat().st_size > 0
