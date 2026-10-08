"""
Renders a preview per Bench2Drive route for choosing the city / country / highway scenarios:
a top-down view with the route (green), start (blue), scenario trigger (red) and other-actor location (orange),
next to a driver's-eye view at the trigger looking along the route. Also writes an index.md gallery.

Usage (no CARLA server needed, it starts one on --port):
  python friction_ood/scenario_selection/preview_routes.py --routes 17752 24252 3540 3813 --out results/scenario_selection
"""
import argparse
import math
import os
import queue
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

import carla
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
B2D_ROUTES = os.path.join(GARAGE, 'Bench2Drive', 'leaderboard', 'data', 'bench2drive220.xml')
TOP_PX, VIEW_W, VIEW_H = 1024, 1024, 576


def speed_limit(town, xyz):
  data = np.load(os.path.join(GARAGE, 'team_code', 'speed_limits', f'{town}_speed_limits.npy'), allow_pickle=True).item()
  return float(data['speed_limits'][cKDTree(data['locations']).query(xyz)[1]])


def load_routes(ids):
  routes = []
  for r in ET.parse(B2D_ROUTES).getroot().findall('route'):
    if r.get('id') not in ids:
      continue
    s = r.find('scenarios')[0]
    xyz = lambda e: np.array([float(e.get(k)) for k in 'xyz'])
    other = s.find('other_actor_location')
    routes.append({
        'id': r.get('id'), 'town': r.get('town'), 'scenario': s.get('type'),
        'waypoints': np.array([xyz(p) for p in r.find('waypoints')]),
        'trigger': xyz(s.find('trigger_point')), 'trigger_yaw': float(s.find('trigger_point').get('yaw', 0.0)),
        'other': xyz(other) if other is not None else None,
    })
  return routes


def capture(world, bp, transform):
  cam = world.spawn_actor(bp, transform)
  images = queue.Queue()
  cam.listen(images.put)
  for _ in range(30):  # let streaming / exposure settle
    world.tick()
  image = None
  while not images.empty():
    image = images.get()
  cam.destroy()
  arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2::-1]
  return Image.fromarray(arr.copy())


def render_route(world, route, out_dir):
  bl = world.get_blueprint_library()
  wps = route['waypoints']
  # A hero near the route makes large maps stream the tiles there.
  hero_bp = bl.find('vehicle.lincoln.mkz_2020')
  hero_bp.set_attribute('role_name', 'hero')
  start = carla.Location(*wps[0]) + carla.Location(z=1.0)
  hero = world.spawn_actor(hero_bp, carla.Transform(start))
  hero.set_simulate_physics(False)
  for _ in range(60):
    world.tick()

  # Top-down camera centred on the route, high enough to see it all with a margin.
  pts = np.vstack([wps[:, :2], route['trigger'][None, :2]] + ([route['other'][None, :2]] if route['other'] is not None else []))
  center = pts.mean(axis=0)
  extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) + 60.0
  ground_z = float(np.median(wps[:, 2]))
  height = extent / 2.0  # fov 90 -> visible width = 2 * height
  top_bp = bl.find('sensor.camera.rgb')
  for k, v in (('image_size_x', TOP_PX), ('image_size_y', TOP_PX), ('fov', 90)):
    top_bp.set_attribute(k, str(v))
  top = capture(world, top_bp, carla.Transform(carla.Location(center[0], center[1], ground_z + height),
                                               carla.Rotation(pitch=-90.0, yaw=0.0)))
  # Pixel projection for pitch -90, yaw 0: image right = +y world, image up = +x world.
  f = TOP_PX / 2.0

  def to_px(p):
    return (TOP_PX / 2 + (p[1] - center[1]) * f / height, TOP_PX / 2 - (p[0] - center[0]) * f / height)

  draw = ImageDraw.Draw(top)
  draw.line([to_px(p) for p in wps], fill=(0, 230, 0), width=6)
  for p, color in ((wps[0], (40, 90, 255)), (route['trigger'], (255, 30, 30)), (route['other'], (255, 150, 0))):
    if p is not None:
      x, y = to_px(p)
      draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=color, outline=(0, 0, 0), width=3)
  scale = 50.0 * f / height  # 50 m scale bar
  draw.rectangle([20, TOP_PX - 40, 20 + scale, TOP_PX - 30], fill=(255, 255, 255), outline=(0, 0, 0))
  draw.text((20, TOP_PX - 60), '50 m', fill=(255, 255, 255))

  # Driver's-eye view: 8 m before the trigger, looking along the route.
  i = int(np.argmin(np.linalg.norm(wps[:, :2] - route['trigger'][:2], axis=1)))
  j = min(i + 3, len(wps) - 1)
  heading = math.degrees(math.atan2(wps[j, 1] - wps[i, 1], wps[j, 0] - wps[i, 0])) if j > i else route['trigger_yaw']
  eye = carla.Location(*route['trigger']) - 8.0 * carla.Location(math.cos(math.radians(heading)),
                                                                  math.sin(math.radians(heading)), 0.0)
  view_bp = bl.find('sensor.camera.rgb')
  for k, v in (('image_size_x', VIEW_W), ('image_size_y', VIEW_H), ('fov', 90)):
    view_bp.set_attribute(k, str(v))
  # The streaming hero sits at the route start, often where the eye is; move it out of the view first.
  hero.set_transform(carla.Transform(start + carla.Location(z=200.0)))
  world.tick()
  view = capture(world, view_bp, carla.Transform(eye + carla.Location(z=2.0), carla.Rotation(pitch=-5.0, yaw=heading)))
  hero.destroy()
  world.tick()

  length = float(np.sum(np.linalg.norm(np.diff(wps[:, :2], axis=0), axis=1)))
  limit = speed_limit(route['town'], route['trigger'])
  canvas = Image.new('RGB', (TOP_PX + VIEW_W, TOP_PX), (25, 25, 25))
  canvas.paste(top, (0, 0))
  canvas.paste(view, (TOP_PX, 0))
  font = ImageFont.load_default()
  info = [f"Route {route['id']}  |  {route['scenario']}  |  {route['town']}",
          f'Speed limit at trigger: {limit:.0f} km/h   |   route length: {length:.0f} m',
          'Top-down: green = route, blue = start, red = scenario trigger, orange = other actor',
          "Right: driver's view 8 m before the trigger, looking along the route"]
  ImageDraw.Draw(canvas).multiline_text((TOP_PX + 20, VIEW_H + 30), '\n'.join(info), fill=(235, 235, 235),
                                        font=font, spacing=12)
  name = f"{route['id']}_{route['scenario']}_{route['town']}.png"
  canvas.save(os.path.join(out_dir, name))
  print(f"  wrote {name} (limit {limit:.0f} km/h, {length:.0f} m)", flush=True)
  return {'file': name, 'limit': limit, 'length': length, **{k: route[k] for k in ('id', 'scenario', 'town')}}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--routes', nargs='+', required=True, help='Bench2Drive route ids')
  parser.add_argument('--out', required=True)
  parser.add_argument('--port', type=int, default=2000)
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)

  env = dict(os.environ, VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
  server = subprocess.Popen([os.path.join(GARAGE, 'carla', 'CarlaUE4.sh'), '-RenderOffScreen', '-nosound',
                             f'-carla-rpc-port={args.port}'], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
  results = []
  try:
    # A CARLA client call made before the server listens blocks for the whole timeout instead of failing,
    # so wait for the port first and probe with a short timeout.
    for _ in range(120):
      with socket.socket() as s:
        if s.connect_ex(('localhost', args.port)) == 0:
          break
      time.sleep(2)
    client = carla.Client('localhost', args.port)
    client.set_timeout(20.0)
    for _ in range(30):
      try:
        client.get_world()
        break
      except RuntimeError:
        time.sleep(5)
    client.set_timeout(600.0)  # large maps take minutes to load
    print('connected to CARLA', flush=True)
    routes = load_routes(set(args.routes))
    for town in sorted({r['town'] for r in routes}):
      print(f'loading {town}', flush=True)
      world = client.load_world(town)
      settings = world.get_settings()
      settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
      world.apply_settings(settings)
      world.set_weather(carla.WeatherParameters.ClearNoon)
      for route in [r for r in routes if r['town'] == town]:
        results.append(render_route(world, route, args.out))
      settings.synchronous_mode = False
      world.apply_settings(settings)
  finally:
    os.killpg(server.pid, 9)

  with open(os.path.join(args.out, 'index.md'), 'w', encoding='utf-8') as f:
    f.write('# Candidate routes\n\n| Route | Scenario | Town | Speed limit at trigger | Length |\n|---|---|---|---|---|\n')
    for r in sorted(results, key=lambda r: r['limit']):
      f.write(f"| {r['id']} | {r['scenario']} | {r['town']} | {r['limit']:.0f} km/h | {r['length']:.0f} m |\n")
    for r in sorted(results, key=lambda r: r['limit']):
      f.write(f"\n## {r['id']} {r['scenario']} ({r['town']}, {r['limit']:.0f} km/h)\n\n![{r['id']}]({r['file']})\n")
  print('done', flush=True)


if __name__ == '__main__':
  main()
