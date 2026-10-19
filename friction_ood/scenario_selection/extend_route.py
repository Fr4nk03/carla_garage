"""
Prepends an approach to a Bench2Drive / Leaderboard route so the ego reaches its cruising speed before the scenario.

The route's first waypoint is projected onto the driving lane of the town's OpenDRIVE map (loaded client-side, no
simulator needed) and the lane is followed upstream with waypoint.previous(); at a split the predecessor with the
smallest heading change is kept. If the extension would start inside a junction it walks further back until clear.
No waypoints are kept inside junctions: there, lanes of the through road and of ramps overlap, a keypoint can snap to
the wrong one and the leaderboard's route planner then plans a long detour (seen: a 5 m hop became 4.9 km). The
leaderboard bridges the junction itself. As a check, the route is expanded offline with the same GlobalRoutePlanner
the leaderboard uses; the script refuses to write it if the planned path is much longer than expected.
Only <waypoints> changes: the scenario block (trigger point and parameters), weather and route id are untouched, and
the leaderboard re-densifies the path with its route planner anyway.

Writes <out>.xml and <out>.png (road network, planned path, original route, extension, trigger, other actor).

Usage:
  python friction_ood/scenario_selection/extend_route.py friction_ood/routes/route_3080.xml --extend 600 \
      --out friction_ood/routes/route_3080_ext600.xml
"""
import argparse
import glob
import math
import os
import xml.etree.ElementTree as ET

import carla
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from scipy.spatial import cKDTree  # pylint: disable=wrong-import-position

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MAPS_DIR = os.path.join(GARAGE, 'carla', 'CarlaUE4', 'Content', 'Carla', 'Maps')
sys.path.insert(0, os.path.join(GARAGE, 'carla', 'PythonAPI', 'carla'))
from agents.navigation.global_route_planner import GlobalRoutePlanner  # pylint: disable=wrong-import-position


def load_map(town):
  """Small towns keep their OpenDRIVE file in Maps/OpenDrive, large maps (Town11-15) in Maps/<Town>/OpenDrive."""
  hits = glob.glob(os.path.join(MAPS_DIR, '**', f'{town}.xodr'), recursive=True)
  if not hits:
    raise FileNotFoundError(f'{town}.xodr not found under {MAPS_DIR}')
  with open(hits[0], encoding='utf-8') as f:
    return carla.Map(town, f.read())


def yaw_diff(a, b):
  return abs((a - b + 180.0) % 360.0 - 180.0)


def walk_upstream(start_wp, extend_m, step):
  """Waypoints upstream of start_wp (nearest first), at least extend_m long and not starting inside a junction."""
  wp, back, travelled = start_wp, [], 0.0
  while travelled < extend_m or wp.is_junction:
    candidates = wp.previous(step)
    if not candidates:
      print(f'  road ends after {travelled:.0f} m upstream; extension shortened')
      break
    wp = min(candidates, key=lambda c: yaw_diff(c.transform.rotation.yaw, wp.transform.rotation.yaw))
    back.append(wp)
    travelled += step
    if travelled > extend_m + 500:  # safety stop if a junction region never ends
      break
  while back and back[-1].is_junction:  # never spawn inside a junction
    back.pop()
  return back


def speed_limits(town, xy):
  path = os.path.join(GARAGE, 'team_code', 'speed_limits', f'{town}_speed_limits.npy')
  if not os.path.exists(path):
    return None, None
  data = np.load(path, allow_pickle=True).item()
  tree = cKDTree(data['locations'][:, :2])
  return data['speed_limits'][tree.query(xy)[1]], data['locations'][:, :2]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('routes', help='route xml (all routes in it are extended, or only --route-id)')
  parser.add_argument('--extend', type=float, default=600.0, help='metres of approach to prepend')
  parser.add_argument('--step', type=float, default=5.0, help='spacing of the new waypoints (m)')
  parser.add_argument('--route-id', default=None)
  parser.add_argument('--max-detour', type=float, default=1.2, help='max planned / polyline length ratio')
  parser.add_argument('--out', required=True, help='output route xml (a .png with the same stem is written too)')
  args = parser.parse_args()

  tree = ET.parse(args.routes)
  root = tree.getroot()
  routes = [r for r in root.findall('route') if args.route_id is None or r.get('id') == args.route_id]
  if not routes:
    raise SystemExit('no matching route')
  maps, planners = {}, {}
  for route in routes:
    town = route.get('town')
    cmap = maps.setdefault(town, load_map(town))
    wps_xml = route.find('waypoints')
    orig = np.array([[float(p.get(k)) for k in 'xyz'] for p in wps_xml.findall('position')])
    start_wp = cmap.get_waypoint(carla.Location(*orig[0]), project_to_road=True, lane_type=carla.LaneType.Driving)
    print(f'route {route.get("id")} ({town}): original start {orig[0][:2].round(1)}, '
          f'lane {start_wp.road_id}/{start_wp.lane_id}, heading {start_wp.transform.rotation.yaw:.1f} deg')

    back = walk_upstream(start_wp, args.extend, args.step)
    junctions = sum(w.is_junction for w in back)
    extended_m = len(back) * args.step
    back = [w for w in back if not w.is_junction]  # see module docstring
    ext = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in back])
    for loc in ext:  # prepend in driving order: the farthest point becomes the new start
      wps_xml.insert(0, ET.Element('position', x=f'{loc[0]:.1f}', y=f'{loc[1]:.1f}', z=f'{loc[2]:.1f}'))

    turn = yaw_diff(back[-1].transform.rotation.yaw, start_wp.transform.rotation.yaw) if back else 0.0
    limits, network = speed_limits(town, ext[:, :2]) if len(ext) else (None, None)
    print(f'  extended by {extended_m:.0f} m (new start {ext[-1][:2].round(1) if len(ext) else orig[0][:2]}); '
          f'{len(back)} waypoints kept, {junctions} inside junctions dropped; heading change start->new start {turn:.0f} deg')
    # Expand the whole route the way the leaderboard does and compare with the expected length.
    planner = planners.setdefault(town, GlobalRoutePlanner(cmap, 1.0))
    keypoints = np.vstack([ext[::-1]] + [orig]) if len(ext) else orig
    planned = []
    for a, b in zip(keypoints[:-1], keypoints[1:]):
      planned += [(w.transform.location.x, w.transform.location.y) for w, _ in planner.trace_route(carla.Location(*a), carla.Location(*b))]
    planned = np.array(planned)
    planned_len = float(np.sum(np.linalg.norm(np.diff(planned, axis=0), axis=1)))
    expected_len = float(np.sum(np.linalg.norm(np.diff(keypoints[:, :2], axis=0), axis=1)))
    print(f'  planned path {planned_len:.0f} m vs keypoint polyline {expected_len:.0f} m')
    if planned_len > args.max_detour * expected_len + 50:
      raise SystemExit(f'  ERROR: the route planner makes a detour ({planned_len:.0f} m); route not written')
    if limits is not None:
      values, counts = np.unique(limits, return_counts=True)
      print('  posted limits along the extension (km/h: share): ' +
            ', '.join(f'{v:.0f}: {100 * c / len(limits):.0f}%' for v, c in zip(values, counts)))

    # Check plot: road network, original route, extension, trigger and other actor.
    scen = route.find('scenarios')
    trig = scen.find('scenario/trigger_point') if scen is not None else None
    other = scen.find('scenario/other_actor_location') if scen is not None else None
    pts = np.vstack([orig[:, :2]] + ([ext[:, :2]] if len(ext) else []))
    lo, hi = pts.min(axis=0) - 80, pts.max(axis=0) + 80
    fig, ax = plt.subplots(figsize=(10, 10))
    if network is not None:
      sel = np.all((network >= lo) & (network <= hi), axis=1)
      ax.scatter(network[sel, 0], network[sel, 1], s=0.3, c='#c8c8c8', linewidths=0, label='road network')
    ax.plot(planned[:, 0], planned[:, 1], c='k', lw=1, ls='--', zorder=4, label=f'planned path ({planned_len:.0f} m)')
    if len(ext):
      ax.scatter(ext[:, 0], ext[:, 1], s=12, c='#ff7f0e', zorder=3, label=f'extension keypoints ({extended_m:.0f} m)')
      ax.scatter(*ext[-1, :2], s=120, c='#1f77b4', zorder=5, label='new start')
    ax.plot(orig[:, 0], orig[:, 1], c='#2ca02c', lw=3, label=f'original route ({len(orig)} keypoints)')
    ax.scatter(*orig[0, :2], s=80, c='#2ca02c', edgecolors='k', zorder=5, label='original start')
    if trig is not None:
      ax.scatter(float(trig.get('x')), float(trig.get('y')), s=140, c='#d62728', marker='*', zorder=6,
                 label='scenario trigger')
    if other is not None:
      ax.scatter(float(other.get('x')), float(other.get('y')), s=100, c='#ff7f0e', edgecolors='k', zorder=6,
                 label='other actor')
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()  # CARLA is left-handed: +y points down in a top view
    ax.set_title(f'Route {route.get("id")} ({town}) extended by {extended_m:.0f} m')
    ax.legend(loc='best')
    fig.tight_layout()
    png = os.path.splitext(args.out)[0] + ('' if len(routes) == 1 else f'_{route.get("id")}') + '.png'
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f'  check plot: {png}')

  tree.write(args.out)
  print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
