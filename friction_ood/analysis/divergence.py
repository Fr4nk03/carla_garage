"""
Plan-vs-reality divergence and driving outcomes from friction_ood tick logs.

For every condition directory (containing simulation_results.json and ticks/*.jsonl.gz) it computes per route:
  outcome   : driving score, route completion, status, infraction counts (from the leaderboard json)
  lateral   : cross-track distance between the actual position at t+h and the path planned at t (h = 0.5, 1, 2 s)
  speed     : actual speed at t+h minus the target speed chosen at t
  braking   : for each stop intent (target speed drops to 0 while moving): distance and time until standstill
  dynamics  : peak sideslip angle, RMS yaw-rate error vs. the TF++ kinematic bicycle model (dry-road reference)
and prints a paired comparison of each low-friction condition against the dry one.

Usage:
  python friction_ood/analysis/divergence.py results/pilot/tfpp --dry ego_mu1.0
  python friction_ood/analysis/divergence.py results/pilot/tfpp --check-frame   # verify the plan's y-axis sign
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'team_code'))
from config import GlobalConfig  # pylint: disable=wrong-import-position

DT = 0.05
HORIZONS = (0.5, 1.0, 2.0)
MOVING = 2.0  # m/s, below this sideslip and yaw-rate errors are meaningless
_CFG = GlobalConfig()


def load_ticks(path):
  meta, rows = {}, []
  with gzip.open(path, 'rt') as f:
    for line in f:
      row = json.loads(line)
      if 'meta' in row:
        meta = row['meta']
      else:
        rows.append(row)
  return meta, rows


def ego_to_world(points, x, y, yaw_deg, y_sign=1.0):
  """Ego frame (x forward, y right) -> CARLA world. CARLA is left-handed: right vector = (-sin, cos)."""
  yaw = math.radians(yaw_deg)
  c, s = math.cos(yaw), math.sin(yaw)
  px, py = points[:, 0], y_sign * points[:, 1]
  return np.stack([x + px * c - py * s, y + px * s + py * c], axis=1)


def point_to_polyline(p, line):
  a, b = line[:-1], line[1:]
  ab = b - a
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
  return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


def kinematic_yaw_rate(speed, steer):
  lf, lr = _CFG.front_wheel_base, _CFG.rear_wheel_base
  beta = math.atan(lr / (lf + lr) * math.tan(_CFG.steering_gain * steer))
  return speed / lr * math.sin(beta)


def route_divergence(rows, y_sign=1.0):
  n = len(rows)
  speed = np.array([math.hypot(r['vx'], r['vy']) for r in rows])
  lat = {h: [] for h in HORIZONS}
  spd = {h: [] for h in HORIZONS}
  plan = None
  for i, r in enumerate(rows):
    plan = r.get('plan', plan)
    if plan is None or 'path' not in plan or speed[i] < MOVING:
      continue
    path = ego_to_world(np.vstack([[0.0, 0.0], np.asarray(plan['path'])]), r['x'], r['y'], r['yaw'], y_sign)
    for h in HORIZONS:
      j = i + int(round(h / DT))
      if j < n:
        lat[h].append(point_to_polyline(np.array([[rows[j]['x'], rows[j]['y']]]), path))
        spd[h].append(speed[j] - plan['target_speed'])

  # Stop intents: target speed switches to 0 while the car is still moving.
  stops, prev_target, plan = [], None, None
  for i, r in enumerate(rows):
    plan = r.get('plan', plan)
    if plan is None or 'target_speed' not in plan:
      continue
    target = plan['target_speed']
    if target == 0.0 and (prev_target or 0.0) > 0.0 and speed[i] > 3.0:
      j = i
      while j < n - 1 and speed[j] > 0.1:
        j += 1
      dist = sum(speed[k] * DT for k in range(i, j))
      stops.append({'v0': float(speed[i]), 'dist': dist, 'time': (j - i) * DT, 'stopped': bool(speed[j] <= 0.1)})
    prev_target = target

  moving = [r for r in rows if r['vx'] > MOVING]
  sideslip = [abs(math.degrees(math.atan2(r['vy'], r['vx']))) for r in moving]
  yaw_err = [r['yaw_rate'] - kinematic_yaw_rate(r['vx'], r['steer']) for r in moving]

  out = {}
  for h in HORIZONS:
    out[f'lat_err_{h}s_mean'] = float(np.mean(lat[h])) if lat[h] else float('nan')
    out[f'lat_err_{h}s_p95'] = float(np.percentile(lat[h], 95)) if lat[h] else float('nan')
    out[f'speed_err_{h}s_mean'] = float(np.mean(spd[h])) if spd[h] else float('nan')
  out['n_stop_intents'] = len(stops)
  out['stop_dist_mean'] = float(np.mean([s['dist'] for s in stops])) if stops else float('nan')
  out['stop_time_mean'] = float(np.mean([s['time'] for s in stops])) if stops else float('nan')
  out['peak_sideslip_deg'] = max(sideslip, default=float('nan'))
  out['yaw_rate_rmse'] = float(np.sqrt(np.mean(np.square(yaw_err)))) if yaw_err else float('nan')
  return out


def load_outcomes(results_json):
  if not os.path.exists(results_json):
    return {}
  records = json.load(open(results_json, encoding='utf-8'))['_checkpoint']['records']
  out = {}
  for rec in records:
    infractions = {k: len(v) for k, v in rec['infractions'].items() if v}
    out[rec['route_id']] = {
        'scenario': rec['scenario_name'],
        'status': rec['status'],
        'driving_score': rec['scores']['score_composed'],
        'route_completion': rec['scores']['score_route'],
        'collisions': sum(n for k, n in infractions.items() if k.startswith('collisions')),
        'infractions': json.dumps(infractions),
    }
  return out


def analyze_condition(cond_dir, y_sign=1.0):
  outcomes = load_outcomes(os.path.join(cond_dir, 'simulation_results.json'))
  per_route = {}
  for path in sorted(glob.glob(os.path.join(cond_dir, 'ticks', '*.jsonl.gz'))):
    meta, rows = load_ticks(path)
    if rows:
      per_route[meta['route']] = {**outcomes.get(meta['route'], {}), **route_divergence(rows, y_sign)}
  return per_route


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='e.g. results/pilot/tfpp (contains <mode>_mu<scale>/seed<k>/)')
  parser.add_argument('--dry', default='ego_mu1.0', help='condition name used as the paired reference')
  parser.add_argument('--seed', default='seed0')
  parser.add_argument('--y-sign', type=float, default=1.0, help='+1 if the plan uses y = right')
  parser.add_argument('--check-frame', action='store_true', help='compare lateral error for both y-axis signs')
  args = parser.parse_args()

  # Folders starting with '_' hold superseded / invalid runs and are skipped.
  conditions = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(os.path.join(args.model_dir, '[!_]*', args.seed)))

  if args.check_frame:
    dry_dir = os.path.join(args.model_dir, args.dry, args.seed)
    for sign in (1.0, -1.0):
      res = analyze_condition(dry_dir, sign)
      print(f'y_sign={sign:+.0f}: mean 0.5 s lateral error on dry =',
            np.nanmean([r['lat_err_0.5s_mean'] for r in res.values()]))
    print('The sign with the much smaller error is the plan frame; pass it as --y-sign.')
    return

  results = {c: analyze_condition(os.path.join(args.model_dir, c, args.seed), args.y_sign) for c in conditions}
  for cond, per_route in results.items():
    out_csv = os.path.join(args.model_dir, cond, args.seed, 'divergence.csv')
    keys = sorted({k for r in per_route.values() for k in r})
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=['route'] + keys)
      writer.writeheader()
      for route, rec in per_route.items():
        writer.writerow({'route': route, **rec})
    print(f'wrote {out_csv} ({len(per_route)} routes)')

  dry = results.get(args.dry, {})
  metrics = ['driving_score', 'route_completion', 'collisions', 'lat_err_1.0s_mean', 'lat_err_2.0s_p95',
             'speed_err_1.0s_mean', 'stop_dist_mean', 'peak_sideslip_deg', 'yaw_rate_rmse']
  for cond, per_route in results.items():
    if cond == args.dry:
      continue
    routes = sorted(set(per_route) & set(dry))
    print(f'\n=== {cond} vs {args.dry} (paired over {len(routes)} routes) ===')
    print(f'{"metric":24} {"dry":>9} {cond:>12} {"delta":>9}')
    for m in metrics:
      a = np.array([dry[r].get(m, np.nan) for r in routes], dtype=float)
      b = np.array([per_route[r].get(m, np.nan) for r in routes], dtype=float)
      print(f'{m:24} {np.nanmean(a):9.2f} {np.nanmean(b):12.2f} {np.nanmean(b - a):9.2f}')
    print('per route (scenario: DS dry -> low, status):')
    for r in routes:
      print(f'  {r:28} {dry[r].get("scenario", "?"):32} {dry[r].get("driving_score", np.nan):6.1f} -> '
            f'{per_route[r].get("driving_score", np.nan):6.1f}  {per_route[r].get("status", "?")}  '
            f'{per_route[r].get("infractions", "")}')


if __name__ == '__main__':
  main()
