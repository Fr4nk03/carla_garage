"""
Road-type / speed coverage of a CARLA route benchmark, without a simulator.

Each route's keypoints are expanded into the driven path with CARLA's GlobalRoutePlanner on the town's OpenDRIVE
map (loaded client-side), as the leaderboard does. Every path point gets the posted speed limit from PDM-Lite's
speed-limit maps (team_code/speed_limits). Outputs per benchmark:
  <out>/<town>.png      town road network (grey) with the benchmark's routes coloured by posted limit
  <out>/summary.png     share of route length per speed class, per town and overall
  <out>/samples.csv     sample path points per (town, speed class) for camera snapshots
  <out>/coverage.md     table of the numbers above

Usage:
  python friction_ood/scenario_selection/benchmark_coverage.py leaderboard/data/longest6.xml --name "Longest6 v2" \
      --out ../results/benchmark_coverage/longest6
"""
import argparse
import collections
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET

import carla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from scipy.spatial import cKDTree  # pylint: disable=wrong-import-position

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(GARAGE, 'carla', 'PythonAPI', 'carla'))
from agents.navigation.global_route_planner import GlobalRoutePlanner  # pylint: disable=wrong-import-position

MAPS_DIR = os.path.join(GARAGE, 'carla', 'CarlaUE4', 'Content', 'Carla', 'Maps')


def find_xodr(name):
  """Small towns keep their OpenDRIVE file in Maps/OpenDrive, large maps (Town11-15) in their own folders."""
  hits = [p for p in glob.glob(os.path.join(MAPS_DIR, '**', f'{name}.xodr'), recursive=True)]
  if not hits:
    raise FileNotFoundError(f'{name}.xodr not found under {MAPS_DIR}')
  return hits[0]
CLASSES = [  # (label, test on posted limit km/h, colour)
    ('<= 50 km/h (urban)', lambda v: v <= 50, '#1f77b4'),
    ('60-90 km/h (rural / arterial)', lambda v: 60 <= v <= 90, '#2ca02c'),
    ('100 km/h (rural highway / arterial)', lambda v: v == 100, '#ff7f0e'),
    ('>= 110 km/h (motorway)', lambda v: v >= 110, '#d62728'),
]


def speed_class(v):
  return next(i for i, (_, test, _) in enumerate(CLASSES) if test(v))


class Town:

  def __init__(self, name):
    with open(find_xodr(name), encoding='utf-8') as f:
      self.map = carla.Map(name, f.read())
    self.planner = GlobalRoutePlanner(self.map, 2.0)
    data = np.load(os.path.join(GARAGE, 'team_code', 'speed_limits', f'{name}_speed_limits.npy'),
                   allow_pickle=True).item()
    self.locations, self.limits = data['locations'][:, :2], data['speed_limits']
    self.tree = cKDTree(self.locations)

  def path(self, keypoints):
    """Dense driven path (N x 3: x, y, yaw) through the keypoints, like the leaderboard's interpolation."""
    pts = []
    for a, b in zip(keypoints[:-1], keypoints[1:]):
      for wp, _ in self.planner.trace_route(carla.Location(*a), carla.Location(*b)):
        t = wp.transform
        pts.append((t.location.x, t.location.y, t.rotation.yaw))
    return np.array(pts)

  def limit(self, xy):
    return self.limits[self.tree.query(xy)[1]]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('routes', help='route xml')
  parser.add_argument('--name', required=True, help='benchmark name for titles')
  parser.add_argument('--out', required=True)
  parser.add_argument('--samples-per-class', type=int, default=3)
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)

  by_town = collections.defaultdict(list)
  for r in ET.parse(args.routes).getroot().findall('route'):
    by_town[r.get('town')].append((r.get('id'), np.array([[float(p.get(k)) for k in 'xyz'] for p in r.find('waypoints')])))

  share = {}  # town -> per-class length (m)
  samples = []
  rng = np.random.default_rng(0)
  for town_name in sorted(by_town):
    print(f'{town_name}: {len(by_town[town_name])} routes', flush=True)
    town = Town(town_name)
    lengths = np.zeros(len(CLASSES))
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(town.locations[::4, 0], town.locations[::4, 1], s=0.2, c='#c8c8c8', linewidths=0)
    class_points = collections.defaultdict(list)
    for _, keypoints in by_town[town_name]:
      path = town.path(keypoints)
      if len(path) < 2:
        continue
      cls = np.array([speed_class(v) for v in town.limit(path[:, :2])])
      seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
      for c in range(len(CLASSES)):
        lengths[c] += seg[cls[:-1] == c].sum()
        idx = np.nonzero(cls == c)[0]
        class_points[c] += [path[i] for i in idx]
        if len(idx):
          ax.scatter(path[idx, 0], path[idx, 1], s=3, c=CLASSES[c][2], linewidths=0)
    for c, pts in class_points.items():
      if pts:
        pick = rng.choice(len(pts), size=min(args.samples_per_class, len(pts)), replace=False)
        samples += [(town_name, CLASSES[c][0], *pts[i]) for i in pick]
    share[town_name] = lengths
    total = lengths.sum()
    for c, (label, _, color) in enumerate(CLASSES):
      ax.scatter([], [], s=40, c=color, label=f'{label}: {100 * lengths[c] / total:.0f}%')
    ax.scatter([], [], s=40, c='#c8c8c8', label='road network (not on any route)')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # CARLA is left-handed: +y points down in a top view
    ax.set_title(f'{args.name}: {town_name}, {len(by_town[town_name])} routes, {total / 1000:.1f} km')
    ax.legend(loc='upper right', markerscale=1.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'{town_name}.png'), dpi=110)
    plt.close(fig)

  # Summary bar chart: share of route length per speed class, per town and overall.
  towns = sorted(share)
  rows = towns + ['All']
  data = np.array([share[t] for t in towns] + [sum(share.values())])
  pct = 100 * data / data.sum(axis=1, keepdims=True)
  fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6))
  left = np.zeros(len(rows))
  for c, (label, _, color) in enumerate(CLASSES):
    ax.barh(rows, pct[:, c], left=left, color=color, label=label)
    left += pct[:, c]
  ax.invert_yaxis()
  ax.set_xlabel('share of route length (%)')
  ax.set_xlim(0, 100)
  ax.set_title(f'{args.name}: posted speed limit along the routes')
  ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'summary.png'), dpi=110)
  plt.close(fig)

  with open(os.path.join(args.out, 'samples.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['town', 'speed_class', 'x', 'y', 'yaw'])
    w.writerows(samples)
  with open(os.path.join(args.out, 'coverage.md'), 'w', encoding='utf-8') as f:
    f.write(f'# {args.name}: posted speed limit along the routes\n\n')
    f.write('Share of route length per class (paths expanded with CARLA\'s route planner; limits from PDM-Lite maps). '
            'Posted limits are a proxy for road type.\n\n')
    f.write('| Town | Length (km) | ' + ' | '.join(label for label, _, _ in CLASSES) + ' |\n')
    f.write('|---|---|' + '---|' * len(CLASSES) + '\n')
    for name, d, p in zip(rows, data, pct):
      f.write(f'| {name} | {d.sum() / 1000:.1f} | ' + ' | '.join(f'{v:.0f}%' for v in p) + ' |\n')
    f.write('\n![summary](summary.png)\n')
    for t in towns:
      f.write(f'\n## {t}\n\n![{t}]({t}.png)\n')
  print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
  main()
