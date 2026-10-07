"""
Maps CARLA friction scale -> effective friction coefficient (mu_eff = a / g).

For every (mode, scale) it runs, from 50 km/h on a long straight:
  brake : full brake, measures mean/peak deceleration and stopping distance
  steer : step steer for 2.5 s, measures peak lateral acceleration and sideslip

Usage (CARLA server must be running):
  python friction_ood/calibration/calibrate_friction.py --port 2000 --out friction_ood/calibration/outputs
"""
import argparse
import csv
import math
import os
import sys

import carla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from friction_ood.friction import apply_friction  # pylint: disable=wrong-import-position

G = 9.81
DT = 0.05  # leaderboard runs at 20 Hz
EGO_BP = 'vehicle.lincoln.mkz_2020'
SMOOTH = 10  # 0.5 s moving average to remove contact/step transients from peak values


def find_straight_spawn(world, length=350.0, max_yaw_dev=2.0):
  """Returns the spawn point with the longest straight road ahead (needed for low-mu stopping distances)."""
  carla_map = world.get_map()
  best, best_len = None, -1.0
  for sp in carla_map.get_spawn_points():
    wp = carla_map.get_waypoint(sp.location)
    yaw0, dist = wp.transform.rotation.yaw, 0.0
    while dist < length:
      nxt = wp.next(5.0)
      if not nxt or abs((nxt[0].transform.rotation.yaw - yaw0 + 180) % 360 - 180) > max_yaw_dev:
        break
      wp, dist = nxt[0], dist + 5.0
    if dist > best_len:
      best, best_len = sp, dist
  print(f'Straight spawn: {best.location} with {best_len:.0f} m straight road')
  return best


def local_frame(vehicle):
  """Velocity / acceleration in the vehicle frame (x forward, y right)."""
  tf = vehicle.get_transform()
  fwd, right = tf.get_forward_vector(), tf.get_right_vector()
  v, a = vehicle.get_velocity(), vehicle.get_acceleration()
  dot = lambda p, q: p.x * q.x + p.y * q.y + p.z * q.z
  return dot(v, fwd), dot(v, right), dot(a, fwd), dot(a, right)


def run_trial(world, spawn, mode, scale, test, speed_kmh):
  ego = world.spawn_actor(world.get_blueprint_library().find(EGO_BP), spawn)
  rows = []
  try:
    for _ in range(20):  # let the car settle on the ground
      world.tick()
    # Same code path as the leaderboard hook: friction is applied after the ego exists.
    info = apply_friction(world, ego, mode, scale)
    world.tick()
    target = speed_kmh / 3.6
    ego.enable_constant_velocity(carla.Vector3D(target, 0.0, 0.0))
    for _ in range(60):
      world.tick()
    ego.disable_constant_velocity()
    start = ego.get_location()

    for step in range(int(20.0 / DT)):
      t = step * DT
      if test == 'brake':
        ctrl = carla.VehicleControl(throttle=0.0, brake=1.0)
      else:
        ctrl = carla.VehicleControl(throttle=0.4, steer=0.4 if t < 2.5 else 0.0)
      ego.apply_control(ctrl)
      world.tick()
      vx, vy, ax, ay = local_frame(ego)
      rows.append({
          't': t, 'vx': vx, 'vy': vy, 'ax': ax, 'ay': ay,
          'yaw_rate': ego.get_angular_velocity().z,
          'dist': ego.get_location().distance(start),
      })
      if test == 'brake' and math.hypot(vx, vy) < 0.1:
        break
      if test == 'steer' and t > 3.0:
        break
  finally:
    ego.destroy()
    for trig in world.get_actors().filter('static.trigger.friction'):
      trig.destroy()
    world.tick()
  return info, rows


def smooth(values, k=SMOOTH):
  return [sum(values[i:i + k]) / k for i in range(max(len(values) - k + 1, 1))] if values else [0.0]


def summarize(test, rows):
  if test == 'brake':
    decel = [-r['ax'] for r in rows if 1.0 < math.hypot(r['vx'], r['vy'])]
    mean_decel = sum(decel) / max(len(decel), 1)
    return {
        'mean_decel': mean_decel,
        'peak_decel': max(smooth(decel)),
        'stop_dist': rows[-1]['dist'],
        'stopped': math.hypot(rows[-1]['vx'], rows[-1]['vy']) < 0.1,
        'mu_eff': mean_decel / G,
    }
  lat = smooth([abs(r['ay']) for r in rows])
  return {
      'peak_lat_acc': max(lat),
      'peak_sideslip_deg': max(math.degrees(math.atan2(abs(r['vy']), max(r['vx'], 0.1))) for r in rows),
      'mu_eff_lat': max(lat) / G,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--host', default='localhost')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--town', default='Town06')
  parser.add_argument('--modes', nargs='+', default=['ego', 'road'])
  parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 0.5, 0.3, 0.15, 0.1, 0.05, 0.02])
  parser.add_argument('--speed', type=float, default=50.0, help='initial speed in km/h')
  parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'outputs'))
  args = parser.parse_args()

  client = carla.Client(args.host, args.port)
  client.set_timeout(120.0)
  world = client.load_world(args.town)
  settings = world.get_settings()
  settings.synchronous_mode, settings.fixed_delta_seconds = True, DT
  world.apply_settings(settings)

  os.makedirs(args.out, exist_ok=True)
  spawn = find_straight_spawn(world)
  spawn.location.z += 0.5
  summary = []
  try:
    for mode in args.modes:
      for scale in args.scales:
        for test in ('brake', 'steer'):
          info, rows = run_trial(world, spawn, mode, scale, test, args.speed)
          with open(os.path.join(args.out, f'{mode}_s{scale}_{test}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
          res = {'mode': mode, 'scale': scale, 'test': test, **summarize(test, rows), 'info': info}
          print(res, flush=True)
          summary.append(res)
  finally:
    settings.synchronous_mode, settings.fixed_delta_seconds = False, None
    world.apply_settings(settings)

  keys = sorted({k for r in summary for k in r})
  with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(summary)


if __name__ == '__main__':
  main()
