"""Tests for spatial_indices and citycat_exposure using a stand-in for CityCAT.

The real code calls ``citycat.exe`` through the Windows command prompt. Here a
small Python script called ``citycat.exe`` is put on the PATH; it writes a
maximum-depth CSV with the SAME depth in every cell, chosen per rainfall
number, so the expected exposure classes and damage costs can be worked out
by hand. These tests are skipped on Windows.
"""
import os
import stat
import sys
import textwrap

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box

import configure11_functions as cf

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="uses a POSIX stand-in script for citycat.exe")

# Uniform depth (m) written by the stand-in CityCAT for each rainfall number
DEPTHS = {1030: 0.0, 2030: 0.0625, 3030: 0.25, 5030: 0.5, 10030: 2.0}

FAKE_CITYCAT = textwrap.dedent("""\
    #!{python}
    import os, sys
    args = sys.argv[1:]
    r = int(args[args.index('-r') + 1]); c = int(args[args.index('-c') + 1])
    depth = {depths}[r]
    out = os.path.join(os.environ['FAKE_RUN_PATH'], 'R%dC%d_SurfaceMaps' % (r, c))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'R%d_C%d_max_depth.csv' % (r, c)), 'w') as f:
        f.write('XCen,YCen,Depth\\n')
        for x in range(5, 300, 10):
            for y in range(5, 300, 10):
                f.write('%.1f,%.1f,%.4f\\n' % (x, y, depth))
    """)

# Building footprints (10 m grid; areas in m2)
BUILDINGS = [
    (1, "Residential", box(50, 50, 70, 70)),          # 400 m2
    (2, "Retail", box(150, 150, 180, 170)),           # 600 m2
    (3, "Offices", box(200, 50, 220, 80)),            # 600 m2
    (4, "Public Buildings", box(50, 200, 60, 240)),   # 400 m2
    (5, "Other", box(250, 250, 270, 270)),            # no damage curve
]
# The residential curve gives damages with fractions of a pound (e.g. 250.25),
# which checks that totals are stored as whole pounds (truncated).
RES_DD = np.array([[0, 0], [1, 1001], [3, 3000]], dtype=float)
NONRES_DD = np.array([[0, 0, 0, 0], [1, 10, 20, 5], [3, 30, 60, 15]], dtype=float)


@pytest.fixture
def run_path(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "vector").mkdir(parents=True)
    gpd.GeoDataFrame({"UID": [b[0] for b in BUILDINGS], "Type": [b[1] for b in BUILDINGS]},
                     geometry=[b[2] for b in BUILDINGS], crs="EPSG:27700").to_file(
        run / "vector" / "buildings.shp")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "citycat.exe"
    exe.write_text(FAKE_CITYCAT.format(python=sys.executable, depths=DEPTHS))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_RUN_PATH", str(run))
    monkeypatch.chdir(run)
    return run


def _spatial_indices(run):
    csv_file = run / "R1030C1_SurfaceMaps" / "R1030_C1_max_depth.csv"
    return cf.spatial_indices(str(run / "vector" / "buildings.shp"), str(run),
                              1030, 1, str(csv_file))


def test_spatial_indices_links_cells_to_every_building(run_path):
    shp_field, cell_index, buffer_list = _spatial_indices(run_path)
    assert shp_field == "UID"
    assert len(cell_index) == len(buffer_list) > 0
    assert set(buffer_list) == {1, 2, 3, 4, 5}
    # The cell centred at (55, 55) lies inside building 1. Cells are listed
    # x-major on a 30 x 30 grid, so its data-row number is 5 * 30 + 5.
    cells_of_building_1 = {c for c, b in zip(cell_index, buffer_list) if b == 1}
    assert 155 in cells_of_building_1


def test_citycat_exposure_costs_classes_and_damages(run_path):
    shp_field, cell_index, buffer_list = _spatial_indices(run_path)
    rainfall = np.array(sorted(DEPTHS))
    population = np.array([[1, 0, 1], [0, 1, 0]])
    features = ["1\nzone 0 shape", "2\nzone 1 shape a\nzone 1 shape b", "1\nzone 2 shape"]
    zone_cost = np.array([0.5, 1.25, 2.0])

    cost, high, medium, low, damage = cf.citycat_exposure(
        str(run_path / "vector" / "buildings.shp"), 3, len(population), str(run_path),
        rainfall, 2, population, features, zone_cost, "1\nmaster shape",
        shp_field, cell_index, buffer_list, 0, RES_DD, NONRES_DD)

    np.testing.assert_array_equal(cost, [[2.5], [1.25]])
    # Depth 0 and 0.0625 m -> low; 0.25 m -> medium; 0.5 and 2 m -> high
    for row in range(2):
        np.testing.assert_array_equal(low[row], [5, 5, 0, 0, 0])
        np.testing.assert_array_equal(medium[row], [0, 0, 5, 0, 0])
        np.testing.assert_array_equal(high[row], [0, 0, 0, 5, 5])
    # Damage at depth d: residential from RES_DD (250.25, 500.5 and 2000.5 GBP
    # at 0.25, 0.5 and 2 m); per-m2 rates 10*d, 20*d and 5*d for retail,
    # offices and public buildings (areas 600, 600, 400 m2). Totals such as
    # 5250.25 are stored as whole pounds.
    expected = [0, 0, 5250, 10500, 42000]
    np.testing.assert_array_equal(damage, [expected, expected])

    # GreenAreas.txt for the last chromosome = master + zone 1 (1 + 2 features).
    # The code builds this path with a Windows separator: run_path + '\\GreenAreas.txt'.
    green_areas = (str(run_path) + "\\GreenAreas.txt")
    with open(green_areas) as f:
        assert f.read() == "3\nmaster shape\nzone 1 shape a\nzone 1 shape b"
