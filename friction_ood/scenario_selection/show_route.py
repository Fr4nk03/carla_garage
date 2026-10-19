"""
Draws a route into a running CARLA world so it can be inspected by flying the spectator around (WASD/QE + right
mouse button in the CARLA window).

Drawn (debug shapes are visible in the window, not to sensors):
  planned driving path    as expanded by the leaderboard's GlobalRoutePlanner; orange = prepended extension
                          (before the original start, if --original is given), green = rest of the route
  route keypoints         small white dots
  start / trigger / other actor / scenario end   large markers with labels
  posted speed limit      floating text every --limit-every metres (PDM-Lite speed-limit maps)

Usage (start CARLA with a window first, e.g. VK_ICD_FILENAMES=... ./CarlaUE4.sh -windowed -ResX=1600 -ResY=900):
  python friction_ood/scenario_selection/show_route.py friction_ood/routes/route_3080_ext600.xml \
      --original friction_ood/routes/route_3080.xml
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import carla
import numpy as np
from scipy.spatial import cKDTree

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(GARAGE, 'carla', 'PythonAPI', 'carla'))
from agents.navigation.global_route_planner import GlobalRoutePlanner  # pylint: disable=wrong-import-position

GREEN, ORANGE, WHITE = carla.Color(0, 200, 0), carla.Color(255, 140, 0), carla.Color(255, 255, 255)
RED, BLUE, YELLOW = carla.Color(230, 0, 0), carla.Color(0, 90, 255), carla.Color(255, 230, 0)


def xyz(e):
  return carla.Location(float(e.get('x')), float(e.get('y')), float(e.get('z', 0.0)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('routes', help='route xml')
  parser.add_argument('--route-id', default=None)
  parser.add_argument('--original', default=None, help='original route xml: path before its start is drawn orange')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--life', type=float, default=0.0, help='seconds the drawings stay (0 = until the map reloads)')
  parser.add_argument('--limit-every', type=float, default=50.0, help='metres between speed-limit labels (0 = off)')
  args = parser.parse_args()

  route = next(r for r in ET.parse(args.routes).getroot().findall('route')
               if args.route_id is None or r.get('id') == args.route_id)
  town = route.get('town')
  keypoints = [xyz(p) for p in route.find('waypoints').findall('position')]
  orig_start = None
  if args.original:
    orig = next(r for r in ET.parse(args.original).getroot().findall('route') if r.get('id') == route.get('id'))
    orig_start = xyz(orig.find('waypoints').find('position'))

  client = carla.Client('localhost', args.port)
  client.set_timeout(600.0)  # large maps take minutes to load
  world = client.get_world()
  if not world.get_map().name.endswith(town):
    print(f'loading {town} ...', flush=True)
    world = client.load_world(town)
  cmap = world.get_map()

  # Expand the keypoints the way the leaderboard does.
  planner = GlobalRoutePlanner(cmap, 1.0)
  path = []
  for a, b in zip(keypoints[:-1], keypoints[1:]):
    path += [w.transform.location for w, _ in planner.trace_route(a, b)]
  split = 0  # index of the original start along the planned path
  if orig_start is not None:
    split = min(range(len(path)), key=lambda i: path[i].distance(orig_start))
  length = sum(p.distance(q) for p, q in zip(path[:-1], path[1:]))
  print(f'route {route.get("id")} ({town}): {len(keypoints)} keypoints, planned path {length:.0f} m'
        + (f', extension {sum(p.distance(q) for p, q in zip(path[:split], path[1:split + 1])):.0f} m' if split else ''))

  # Large maps only stream detail around a hero vehicle: park one high above the route's middle.
  mid = path[len(path) // 2]
  bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
  bp.set_attribute('role_name', 'hero')
  hero = world.try_spawn_actor(bp, carla.Transform(mid + carla.Location(z=150.0)))
  if hero is not None:
    hero.set_simulate_physics(False)

  debug, up = world.debug, carla.Location(z=0.6)
  for i, (p, q) in enumerate(zip(path[:-1], path[1:])):
    debug.draw_line(p + up, q + up, thickness=0.25, color=ORANGE if i < split else GREEN, life_time=args.life)
  for k in keypoints:
    debug.draw_point(k + up, size=0.08, color=WHITE, life_time=args.life)

  def marker(loc, color, text):
    debug.draw_point(loc + carla.Location(z=1.0), size=0.4, color=color, life_time=args.life)
    debug.draw_line(loc, loc + carla.Location(z=4.0), thickness=0.15, color=color, life_time=args.life)
    debug.draw_string(loc + carla.Location(z=4.5), text, draw_shadow=True, color=color, life_time=args.life)

  marker(path[0], BLUE, 'START')
  if orig_start is not None:
    marker(orig_start, GREEN, 'original start')
  marker(path[-1], WHITE, 'END')
  for scen in route.iter('scenario'):
    trig = scen.find('trigger_point')
    if trig is not None:
      marker(xyz(trig), RED, f'TRIGGER {scen.get("type")}')
    for tag, color, text in (('other_actor_location', ORANGE, 'other actor'), ('end_location', YELLOW, 'scenario end')):
      e = scen.find(tag)
      if e is not None:
        marker(xyz(e), color, text)

  limits_file = os.path.join(GARAGE, 'team_code', 'speed_limits', f'{town}_speed_limits.npy')
  if args.limit_every > 0 and os.path.exists(limits_file):
    data = np.load(limits_file, allow_pickle=True).item()
    tree, limits = cKDTree(data['locations'][:, :2]), data['speed_limits']
    travelled, next_label = 0.0, 0.0
    for p, q in zip(path[:-1], path[1:]):
      if travelled >= next_label:
        v = limits[tree.query([p.x, p.y])[1]]
        debug.draw_string(p + carla.Location(z=2.5), f'{v:.0f} km/h', draw_shadow=True, color=YELLOW,
                          life_time=args.life)
        next_label += args.limit_every
      travelled += p.distance(q)

  # Put the spectator behind and above the start, looking along the route.
  ahead = path[min(20, len(path) - 1)]
  yaw = math.degrees(math.atan2(ahead.y - path[0].y, ahead.x - path[0].x))
  back = carla.Location(-25 * math.cos(math.radians(yaw)), -25 * math.sin(math.radians(yaw)), 20.0)
  world.get_spectator().set_transform(carla.Transform(path[0] + back, carla.Rotation(pitch=-30.0, yaw=yaw)))
  print('drawn. Fly with WASD / QE and the right mouse button in the CARLA window.')
  print('The parked hero vehicle (for map streaming) stays in the world until CARLA is closed or the map reloads.')


if __name__ == '__main__':
  main()
