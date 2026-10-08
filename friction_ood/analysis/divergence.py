"""
Plan-vs-reality divergence and driving outcomes from friction_ood tick logs.

For every condition directory (containing simulation_results.json and ticks/*.jsonl.gz) it computes per route:
  outcome   : driving score, route completion, status, infraction counts (from the leaderboard json)
  lateral   : cross-track distance between the actual position and the path planned at t, after travelling
              d = 2.5 / 5 / 7.5 m (and after 0.5 s). Positions beyond the end of the planned path (~10 m for TF++)
              are skipped, otherwise the metric measures overshoot past the endpoint instead of lateral drift.
  speed     : actual speed at t+h minus the target speed chosen at t (h = 0.5, 1, 2 s)
  braking   : for each stop intent (model's P(stop) >= 0.5 while moving > 3 m/s), until the first of:
              a leaderboard collision ('impact'), standstill ('stopped'), or P(stop) staying
              below 0.5 for 0.5 s ('aborted'). Distances are reported for completed stops only, with the speed at
              the intent and the ratio to the dry-physics minimum stopping distance v0^2 / (2 * A_DRY).
  dynamics  : peak sideslip angle, RMS yaw-rate error vs. the TF++ kinematic bicycle model (dry-road reference)
and prints a paired comparison of each low-friction condition against the dry one.

Usage:
  python friction_ood/analysis/divergence.py results/pilot/tfpp --dry ego_mu1.0 [--seed all]
  python friction_ood/analysis/divergence.py results/pilot/tfpp --check-frame   # verify the plan's y-axis sign
"""
import argparse
import csv
import glob
import gzip
import json
import math
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'team_code'))
from config import GlobalConfig  # pylint: disable=wrong-import-position

DT = 0.05
HORIZONS = (0.5, 1.0, 2.0)  # s, speed tracking
LAT_DISTANCES = (2.5, 5.0, 7.5)  # m travelled, lateral deviation (must stay within the ~10 m planned path)
MOVING = 2.0  # m/s, below this sideslip and yaw-rate errors are meaningless
A_DRY = 7.44  # m/s^2, mean full-brake deceleration on dry road (friction_ood/calibration, scale 1.0)
COLLISION_RADIUS = 3.0  # m, a leaderboard collision is assigned to the first tick the ego was this close to it
STOP_PROB = 0.5  # stop intent while the model's probability of target-speed class 0 is at least this
ABORT_TICKS = 10  # a stop intent counts as aborted after P(stop) stayed below STOP_PROB for 0.5 s
_CFG = GlobalConfig()


def load_ticks(path):
  """Returns (meta, rows), or (meta, None) for a log that is still being written (route in progress)."""
  meta, rows = {}, []
  try:
    with gzip.open(path, 'rt') as f:
      for line in f:
        row = json.loads(line)
        if 'meta' in row:
          meta = row['meta']
        else:
          rows.append(row)
  except (EOFError, json.JSONDecodeError):
    return meta, None
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


def polyline_length(line):
  return float(np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)))


def kinematic_yaw_rate(speed, steer):
  lf, lr = _CFG.front_wheel_base, _CFG.rear_wheel_base
  beta = math.atan(lr / (lf + lr) * math.tan(_CFG.steering_gain * steer))
  return speed / lr * math.sin(beta)


def collision_ticks(rows, collision_xy):
  """Index of the first tick at which the ego was within COLLISION_RADIUS of each leaderboard collision."""
  xy = np.array([[r['x'], r['y']] for r in rows])
  ticks = set()
  for c in collision_xy:
    dist = np.linalg.norm(xy - np.asarray(c), axis=1)
    close = np.nonzero(dist <= COLLISION_RADIUS)[0]
    ticks.add(int(close[0]) if len(close) else int(np.argmin(dist)))
  return ticks


def route_divergence(rows, y_sign=1.0, collision_xy=()):
  n = len(rows)
  hits = collision_ticks(rows, collision_xy)
  speed = np.array([math.hypot(r['vx'], r['vy']) for r in rows])
  travelled = np.concatenate([[0.0], np.cumsum(speed[:-1] * DT)])  # distance driven up to each tick
  lat_t, lat_d = [], {d: [] for d in LAT_DISTANCES}
  spd = {h: [] for h in HORIZONS}
  plan = None
  for i, r in enumerate(rows):
    plan = r.get('plan', plan)
    if plan is None or 'path' not in plan or speed[i] < MOVING:
      continue
    path = ego_to_world(np.vstack([[0.0, 0.0], np.asarray(plan['path'])]), r['x'], r['y'], r['yaw'], y_sign)
    reach = polyline_length(path) - 0.5  # stay clear of the endpoint, where the projection would clamp
    for h in HORIZONS:
      j = i + int(round(h / DT))
      if j < n:
        spd[h].append(speed[j] - plan['target_speed'])
        if h == 0.5 and travelled[j] - travelled[i] <= reach:
          lat_t.append(point_to_polyline(np.array([[rows[j]['x'], rows[j]['y']]]), path))
    for d in LAT_DISTANCES:
      j = int(np.searchsorted(travelled, travelled[i] + d))
      if d <= reach and j < n:
        lat_d[d].append(point_to_polyline(np.array([[rows[j]['x'], rows[j]['y']]]), path))

  # Stop intents use the model's stop probability (target-speed class 0), not the target speed itself: with
  # uncertainty weighting the target speed flickers above 0 during a stop while P(stop) stays high.
  p_stop = lambda p: p['target_speed_probs'][0] if p and 'target_speed_probs' in p else None
  stops, plan, i = [], None, 0
  while i < n:
    plan = rows[i].get('plan', plan)
    if p_stop(plan) is None or p_stop(plan) < STOP_PROB or speed[i] <= 3.0:
      i += 1
      continue
    j, outcome, step_plan, low = i, 'unfinished', plan, 0
    while j < n - 1:
      j += 1
      step_plan = rows[j].get('plan', step_plan)
      low = low + 1 if p_stop(step_plan) < STOP_PROB else 0
      if j in hits:
        outcome = 'impact'
      elif speed[j] <= 0.1:
        outcome = 'stopped'
      elif low >= ABORT_TICKS:
        outcome = 'aborted'
      if outcome != 'unfinished':
        break
    v0 = float(speed[i])
    stops.append({'v0': v0, 'dist': float(travelled[j] - travelled[i]), 'time': (j - i) * DT, 'outcome': outcome,
                  'ratio': float(travelled[j] - travelled[i]) / (v0 * v0 / (2 * A_DRY))})
    # Next intent only after this episode ended and the stop probability dropped again.
    i = j + 1
    while i < n and (p_stop(rows[i].get('plan', step_plan)) or 0.0) >= STOP_PROB:
      step_plan = rows[i].get('plan', step_plan)
      i += 1
  done = [s for s in stops if s['outcome'] == 'stopped']

  moving = [r for r in rows if r['vx'] > MOVING]
  sideslip = [abs(math.degrees(math.atan2(r['vy'], r['vx']))) for r in moving]
  yaw_err = [r['yaw_rate'] - kinematic_yaw_rate(r['vx'], r['steer']) for r in moving]

  mean = lambda values: float(np.mean(values)) if values else float('nan')
  p95 = lambda values: float(np.percentile(values, 95)) if values else float('nan')
  out = {'lat_err_0.5s_mean': mean(lat_t), 'lat_err_0.5s_p95': p95(lat_t)}
  for d in LAT_DISTANCES:
    out[f'lat_dev_{d}m_mean'] = mean(lat_d[d])
    out[f'lat_dev_{d}m_p95'] = p95(lat_d[d])
  for h in HORIZONS:
    out[f'speed_err_{h}s_mean'] = mean(spd[h])
  out['n_stop_intents'] = len(stops)
  for outcome in ('stopped', 'impact', 'aborted', 'unfinished'):
    out[f'n_stop_{outcome}'] = sum(s['outcome'] == outcome for s in stops)
  out['stop_v0_mean'] = mean([s['v0'] for s in stops])
  out['stop_dist_mean'] = mean([s['dist'] for s in done])
  out['stop_time_mean'] = mean([s['time'] for s in done])
  out['stop_dist_ratio_mean'] = mean([s['ratio'] for s in done])
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
        '_collision_xy': [(float(m.group(1)), float(m.group(2))) for k, v in rec['infractions'].items()
                          if k.startswith('collisions') for e in v
                          for m in [re.search(r'x=(-?[\d.]+), y=(-?[\d.]+)', e)] if m],
        'infractions': json.dumps(infractions),
    }
  return out


def analyze_condition(cond_dir, y_sign=1.0):
  outcomes = load_outcomes(os.path.join(cond_dir, 'simulation_results.json'))
  per_route = {}
  for path in sorted(glob.glob(os.path.join(cond_dir, 'ticks', '*.jsonl.gz'))):
    meta, rows = load_ticks(path)
    if rows is None:
      print(f'skipping unfinished log {os.path.basename(path)}')
    elif rows:
      outcome = dict(outcomes.get(meta['route'], {}))
      if outcome.get('status', '').startswith('Excluded'):
        print(f"excluding {meta['route']} ({outcome['status']})")
        continue
      collision_xy = outcome.pop('_collision_xy', [])
      per_route[meta['route']] = {**outcome, **route_divergence(rows, y_sign, collision_xy)}
  return per_route


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='e.g. results/pilot/tfpp (contains <mode>_mu<scale>/seed<k>/)')
  parser.add_argument('--dry', default='ego_mu1.0', help='condition name used as the paired reference')
  parser.add_argument('--seed', default='seed0', help="seed folder, or 'all' to pair every (route, seed)")
  parser.add_argument('--y-sign', type=float, default=1.0, help='+1 if the plan uses y = right')
  parser.add_argument('--check-frame', action='store_true', help='compare lateral error for both y-axis signs')
  args = parser.parse_args()

  # Folders starting with '_' hold superseded / invalid runs and are skipped (see below).
  if args.check_frame:
    dry_dir = os.path.join(args.model_dir, args.dry, 'seed0' if args.seed == 'all' else args.seed)
    for sign in (1.0, -1.0):
      res = analyze_condition(dry_dir, sign)
      print(f'y_sign={sign:+.0f}: mean 0.5 s lateral error on dry =',
            np.nanmean([r['lat_err_0.5s_mean'] for r in res.values()]))
    print('The sign with the much smaller error is the plan frame; pass it as --y-sign.')
    return

  # results[cond][seed][route] = metrics. With --seed all, every seed folder that has a finished leaderboard json.
  seeds = ([args.seed] if args.seed != 'all' else
           sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.model_dir, '[!_]*', 'seed*', 'simulation_results.json'))}))
  conditions = sorted({c for c in os.listdir(args.model_dir) if not c.startswith('_')
                       and os.path.isdir(os.path.join(args.model_dir, c))})
  results = {c: {} for c in conditions}
  for cond in conditions:
    for seed in seeds:
      cond_dir = os.path.join(args.model_dir, cond, seed)
      if not os.path.isdir(os.path.join(cond_dir, 'ticks')):
        continue
      per_route = analyze_condition(cond_dir, args.y_sign)
      results[cond][seed] = per_route
      keys = sorted({k for r in per_route.values() for k in r})
      with open(os.path.join(cond_dir, 'divergence.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['route'] + keys)
        writer.writeheader()
        for route, rec in per_route.items():
          writer.writerow({'route': route, **rec})
      print(f'wrote {cond_dir}/divergence.csv ({len(per_route)} routes)')

  dry = results.get(args.dry, {})
  metrics = ['driving_score', 'route_completion', 'collisions', 'lat_err_0.5s_mean', 'lat_dev_2.5m_mean',
             'lat_dev_5.0m_mean', 'lat_dev_7.5m_mean', 'lat_dev_7.5m_p95', 'speed_err_1.0s_mean', 'n_stop_intents',
             'n_stop_stopped', 'n_stop_impact', 'n_stop_aborted', 'stop_v0_mean', 'stop_dist_mean',
             'stop_dist_ratio_mean', 'peak_sideslip_deg', 'yaw_rate_rmse']
  rng = np.random.default_rng(0)
  for cond, by_seed in results.items():
    if cond == args.dry:
      continue
    # Pairs (route, seed) present in both conditions: same route and traffic seed, only friction differs.
    pairs = sorted((r, s) for s in by_seed if s in dry for r in by_seed[s] if r in dry[s])
    routes = sorted({r for r, _ in pairs})
    print(f'\n=== {cond} vs {args.dry} (paired over {len(pairs)} route x seed pairs, {len(routes)} routes, '
          f'seeds {sorted({s for _, s in pairs})}) ===')
    print(f'{"metric":24} {"dry":>9} {cond:>12} {"delta":>9}   {"95% CI (route bootstrap)":>24}')
    summary_rows = []
    for m in metrics:
      a = {(r, s): float(dry[s][r].get(m, np.nan)) for r, s in pairs}
      b = {(r, s): float(by_seed[s][r].get(m, np.nan)) for r, s in pairs}
      # Route-level mean of the paired differences, then resample routes (repetitions of a route are correlated).
      diff = {r: np.nanmean([b[p] - a[p] for p in pairs if p[0] == r]) if any(
          not np.isnan(b[p] - a[p]) for p in pairs if p[0] == r) else np.nan for r in routes}
      d = np.array([v for v in diff.values() if not np.isnan(v)])
      ci = (np.percentile([rng.choice(d, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
            if len(d) > 1 else (np.nan, np.nan))
      row = {'metric': m, 'dry': np.nanmean(list(a.values())), cond: np.nanmean(list(b.values())),
             'delta': d.mean() if len(d) else np.nan, 'ci_low': ci[0], 'ci_high': ci[1], 'n_routes': len(d)}
      summary_rows.append(row)
      print(f"{m:24} {row['dry']:9.2f} {row[cond]:12.2f} {row['delta']:9.2f}   [{ci[0]:8.2f}, {ci[1]:8.2f}]")
    out_csv = os.path.join(args.model_dir, f'paired_{cond}_vs_{args.dry}_{args.seed}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
      writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
      writer.writeheader()
      writer.writerows(summary_rows)
    with open(out_csv.replace('.csv', '_meta.json'), 'w', encoding='utf-8') as f:
      json.dump({'pairs': len(pairs), 'routes': len(routes), 'seeds': sorted({s for _, s in pairs})}, f)
    print(f'wrote {out_csv}')
    print('per route (scenario: DS dry -> low, per seed):')
    for r in routes:
      seeds_r = sorted(s for rr, s in pairs if rr == r)
      scen = dry[seeds_r[0]][r].get('scenario', '?')
      ds = '  '.join(f'{s}: {dry[s][r].get("driving_score", np.nan):5.1f} -> {by_seed[s][r].get("driving_score", np.nan):5.1f}'
                     for s in seeds_r)
      print(f'  {r:28} {scen:30} {ds}')


if __name__ == '__main__':
  main()
