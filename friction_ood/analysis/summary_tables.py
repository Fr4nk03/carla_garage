"""
Renders the per-route divergence.csv of each condition as Markdown tables (one table per condition).

Run divergence.py first. Columns use only the metrics that are valid for TF++'s ~10 m path:
lateral and speed error at the 0.5 s horizon (1 s / 2 s run past the end of the planned path).
Friction re-applies come from <condition>/<seed>/simulation_results_friction.jsonl.

Usage:
  python friction_ood/analysis/summary_tables.py results/pilot/tfpp --out results/pilot/tfpp/summary_tables.md
"""
import argparse
import csv
import glob
import json
import math
import os

# Braking calibration (friction_ood/calibration/outputs/summary.csv): tire friction scale -> mu_eff = a / g.
MU_EFF = {1.0: 0.76, 0.3: 0.52, 0.1: 0.28, 0.05: 0.14, 0.02: 0.06}

COLUMNS = ['Route', 'Scenario', 'Status', 'DS', 'RC', 'Infractions', 'Lat dev 5 m (m)', 'Speed err 0.5 s (m/s)',
           'Stops: done / coll. / aborted', 'Stop dist ÷ dry min', 'Peak sideslip (°)', 'Yaw-rate RMSE (rad/s)',
           'Friction re-applies']


def fmt(value):
  return '—' if math.isnan(float(value)) else f'{float(value):.2f}'


def load_reapplies(cond_dir):
  path = os.path.join(cond_dir, 'simulation_results_friction.jsonl')
  if not os.path.exists(path):
    return {}
  with open(path, encoding='utf-8') as f:
    return {d['route']: len(d.get('reapply_ticks', [])) for d in map(json.loads, f)}


def format_infractions(raw):
  # Bench2Drive no longer counts min-speed infractions in the driving score, so leave them out.
  inf = {k: v for k, v in json.loads(raw).items() if k != 'min_speed_infractions'}
  label = lambda k: k.replace('collisions_', '') + ' collision' if k.startswith('collisions_') else k
  return ', '.join(f'{label(k).replace("_", " ")} ×{v}' for k, v in inf.items()) or '—'


def condition_table(cond_dir, title):
  with open(os.path.join(cond_dir, 'divergence.csv'), encoding='utf-8') as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r['route'].split('_')[1]))
  reapplies = load_reapplies(cond_dir)

  lines = [f'### {title}', '', '| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
  for r in rows:
    lines.append('| ' + ' | '.join([
        r['route'].split('_')[1],
        r['scenario'].rsplit('_', 1)[0],
        r['status'].replace('Failed - ', 'Failed: '),
        f"{float(r['driving_score']):.1f}",
        f"{float(r['route_completion']):.0f}",
        format_infractions(r['infractions']),
        f"{float(r['lat_dev_5.0m_mean']):.2f}",
        f"{float(r['speed_err_0.5s_mean']):+.2f}",
        f"{r['n_stop_stopped']} / {r['n_stop_impact']} / {r['n_stop_aborted']}",
        fmt(r['stop_dist_ratio_mean']),
        f"{float(r['peak_sideslip_deg']):.1f}",
        f"{float(r['yaw_rate_rmse']):.2f}",
        str(reapplies.get(r['route'], '—')),
    ]) + ' |')

  mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)
  total = lambda k: sum(int(r[k]) for r in rows)
  valid = lambda k: [float(r[k]) for r in rows if not math.isnan(float(r[k]))]
  nanmean = lambda k: f'{sum(valid(k)) / len(valid(k)):.2f}' if valid(k) else '—'
  completed = sum(r['status'] == 'Completed' for r in rows)
  lines.append('| ' + ' | '.join([
      '**Mean**', '', f'{completed}/{len(rows)} completed', f"**{mean('driving_score'):.1f}**",
      f"**{mean('route_completion'):.1f}**", f"{mean('collisions'):.1f} collisions/route",
      f"**{mean('lat_dev_5.0m_mean'):.2f}**", f"**{mean('speed_err_0.5s_mean'):+.2f}**",
      f"{total('n_stop_stopped')} / {total('n_stop_impact')} / {total('n_stop_aborted')}", f"**{nanmean('stop_dist_ratio_mean')}**",
      f"**{mean('peak_sideslip_deg'):.1f}**", f"**{mean('yaw_rate_rmse'):.2f}**", ''
  ]) + ' |')
  return '\n'.join(lines)


# Paired-summary metrics shown in the overview, with display names.
PAIRED_METRICS = [
    ('driving_score', 'Driving score'), ('route_completion', 'Route completion (%)'),
    ('collisions', 'Collisions per route'), ('lat_dev_5.0m_mean', 'Lat dev after 5 m (m)'),
    ('lat_dev_7.5m_p95', 'Lat dev after 7.5 m, p95 (m)'), ('speed_err_1.0s_mean', 'Speed err 1 s (m/s)'),
    ('stop_dist_ratio_mean', 'Stop dist ÷ dry min'), ('n_stop_impact', 'Stops ending in a collision'),
    ('peak_sideslip_deg', 'Peak sideslip (°)'), ('yaw_rate_rmse', 'Yaw-rate RMSE (rad/s)'),
]


def paired_table(model_dir, dry):
  """Overview from divergence.py --seed all: paired dry vs. low-friction deltas with route-bootstrap 95% CIs."""
  lines = []
  for path in sorted(glob.glob(os.path.join(model_dir, f'paired_*_vs_{dry}_all.csv'))):
    cond = os.path.basename(path)[len('paired_'):].split('_vs_')[0]
    with open(path, encoding='utf-8') as f:
      rows = {r['metric']: r for r in csv.DictReader(f)}
    with open(path.replace('.csv', '_meta.json'), encoding='utf-8') as f:
      meta = json.load(f)
    scale = float(cond.split('_mu')[1])
    lines += [f'### {cond} vs {dry}: paired over {meta["pairs"]} route × seed pairs '
              f'({meta["routes"]} routes, seeds {", ".join(meta["seeds"])})', '',
              f'| Metric | Dry | Scale {scale}{f" (μ_eff ≈ {MU_EFF[scale]:.2f})" if scale in MU_EFF else ""} | Δ | 95% CI |',
              '|---|---|---|---|---|']
    for key, name in PAIRED_METRICS:
      r = rows[key]
      lines.append(f"| {name} | {fmt(r['dry'])} | {fmt(r[cond])} | **{float(r['delta']):+.2f}** | "
                   f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |")
    lines.append('')
  return '\n'.join(lines)


def per_route_ds_table(model_dir, conds, seeds):
  """Driving score per route and repetition, dry -> low friction."""
  data = {}
  for cond in conds:
    for seed in seeds:
      path = os.path.join(model_dir, cond, seed, 'divergence.csv')
      if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
          for r in csv.DictReader(f):
            data.setdefault(r['route'], {'scenario': r['scenario'].rsplit('_', 1)[0]})[(cond, seed)] = float(
                r['driving_score'])
  lines = ['| Route | Scenario | ' + ' | '.join(f'{s}: dry → low' for s in seeds) + ' | Mean dry → low |',
           '|' + '---|' * (3 + len(seeds))]
  dry, low = conds
  for route in sorted(data, key=lambda r: int(r.split('_')[1])):
    d = data[route]
    cells, pairs = [], []
    for s in seeds:
      if (dry, s) in d and (low, s) in d:
        cells.append(f'{d[(dry, s)]:.1f} → {d[(low, s)]:.1f}')
        pairs.append((d[(dry, s)], d[(low, s)]))
      else:
        cells.append('excluded')
    mean = (f'{sum(p[0] for p in pairs) / len(pairs):.1f} → {sum(p[1] for p in pairs) / len(pairs):.1f}'
            if pairs else '—')
    lines.append(f"| {route.split('_')[1]} | {d['scenario']} | " + ' | '.join(cells) + f' | {mean} |')
  return '\n'.join(lines)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_dir', help='e.g. results/pilot/tfpp (contains <mode>_mu<scale>/<seed>/divergence.csv)')
  parser.add_argument('--seed', default='seed0', help="seed folder, or 'all' for the multi-repetition report")
  parser.add_argument('--dry', default='ego_mu1.0')
  parser.add_argument('--out', default=None, help='Markdown file to write (default: <model_dir>/summary_tables.md)')
  args = parser.parse_args()

  # Folders starting with '_' hold superseded / invalid runs and are skipped.
  seeds = ([args.seed] if args.seed != 'all' else
           sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.model_dir, '[!_]*', 'seed*', 'divergence.csv'))}))
  cond_names = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                       for p in glob.glob(os.path.join(args.model_dir, '[!_]*', 'seed*', 'divergence.csv'))},
                      key=lambda c: -float(c.split('_mu')[1]))  # dry (largest scale) first
  title = os.path.basename(os.path.normpath(args.model_dir))
  sections = [f'# {title}: results ({", ".join(seeds)})', '']
  if args.seed == 'all':
    sections += ['## Paired summary (run `divergence.py --seed all` first)', '',
                 'Each pair is the same route and traffic seed, dry vs. low friction. Δ is the mean over routes of '
                 'the per-route mean paired difference; the CI resamples routes (5000 bootstrap samples).', '',
                 paired_table(args.model_dir, args.dry), '## Driving score per route and repetition', '',
                 per_route_ds_table(args.model_dir, cond_names[:2], seeds), '',
                 '"excluded": no valid pair for that repetition (simulator crash, see the run folder).', '',
                 '## Per-repetition details', '']
  sections += [
              'Generated by `friction_ood/analysis/summary_tables.py` from each condition\'s `divergence.csv`.', '',
              '- **DS / RC**: Bench2Drive driving score / route completion (%).',
              '- **Infractions**: min-speed infractions are omitted (not counted in the B2D driving score).',
              '- **Lat dev 5 m**: distance of the car from the path it planned, after driving 5 m along it (m); '
              'points beyond the end of the ~10 m planned path are skipped.',
              '- **Speed err 0.5 s**: actual speed 0.5 s later minus the chosen target speed (m/s); negative = slower.',
              '- **Peak sideslip**: max |atan(v_lat / v_long)| while moving > 2 m/s.',
              '- **Stops**: stop intents (model P(stop) ≥ 0.5 while > 3 m/s) that ended at standstill / in a '
              'collision / were abandoned (P(stop) < 0.5 for 0.5 s).',
              '- **Stop dist ÷ dry min**: stopping distance of completed stops divided by the dry-physics minimum '
              'v0² / (2 · 7.44 m/s²); ≈ 2.7 is expected at scale 0.1 if the model brakes as hard as it can but no earlier.',
              '- **Yaw-rate RMSE**: actual yaw rate vs. the TF++ kinematic bicycle model (dry-road reference).',
              '- **Friction re-applies**: times the enforcer restored the target tire friction after CARLA reset it.', '']
  for seed in seeds:
    for cond in cond_names:
      cond_dir = os.path.join(args.model_dir, cond, seed)
      if not os.path.exists(os.path.join(cond_dir, 'divergence.csv')):
        continue
      scale = float(cond.split('_mu')[1])
      label = f' (μ_eff ≈ {MU_EFF[scale]:.2f})' if scale in MU_EFF else ''
      sections += [condition_table(cond_dir, f'{cond} / {seed}: tire friction scale {scale}{label}'), '']

  out = args.out or os.path.join(args.model_dir, 'summary_tables.md')
  with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(sections))
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
