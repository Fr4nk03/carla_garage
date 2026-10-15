"""
Driver's-eye snapshots at sample route points (samples.csv from benchmark_coverage.py), to check visually what kind
of road each posted-speed class really is. Writes one contact sheet per benchmark: rows = towns, columns = speed
classes, one image per sample point.

Usage (no CARLA server needed, it starts one on --port; the GPU must be free):
  python friction_ood/scenario_selection/snapshot_samples.py ../results/benchmark_coverage/longest6 --name "Longest6 v2"
"""
import argparse
import csv
import collections
import math
import os
import queue
import socket
import subprocess
import time

import carla
import numpy as np
from PIL import Image, ImageDraw

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
W, H = 480, 270


def start_server(port):
  env = dict(os.environ, VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
  server = subprocess.Popen([os.path.join(GARAGE, 'carla', 'CarlaUE4.sh'), '-RenderOffScreen', '-nosound',
                             f'-carla-rpc-port={port}'], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
  for _ in range(120):  # a client call before the server listens would block for the whole timeout
    with socket.socket() as s:
      if s.connect_ex(('localhost', port)) == 0:
        break
    time.sleep(2)
  return server


def snapshot(world, x, y, yaw):
  bl = world.get_blueprint_library()
  ground = world.get_map().get_waypoint(carla.Location(x, y, 0.0)).transform.location
  # A hero at the point makes large maps stream the tiles there; park it high above, out of the view.
  hero_bp = bl.find('vehicle.lincoln.mkz_2020')
  hero_bp.set_attribute('role_name', 'hero')
  hero = world.spawn_actor(hero_bp, carla.Transform(ground + carla.Location(z=100.0)))
  hero.set_simulate_physics(False)
  for _ in range(40):
    world.tick()
  bp = bl.find('sensor.camera.rgb')
  for k, v in (('image_size_x', W), ('image_size_y', H), ('fov', 90)):
    bp.set_attribute(k, str(v))
  back = carla.Location(-6.0 * math.cos(math.radians(yaw)), -6.0 * math.sin(math.radians(yaw)), 2.0)
  cam = world.spawn_actor(bp, carla.Transform(ground + back, carla.Rotation(pitch=-4.0, yaw=yaw)))
  images = queue.Queue()
  cam.listen(images.put)
  for _ in range(20):
    world.tick()
  image = None
  while not images.empty():
    image = images.get()
  cam.destroy()
  hero.destroy()
  world.tick()
  arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, 2::-1]
  return Image.fromarray(arr.copy())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('coverage_dir', help='output folder of benchmark_coverage.py (contains samples.csv)')
  parser.add_argument('--name', required=True)
  parser.add_argument('--port', type=int, default=2000)
  args = parser.parse_args()

  with open(os.path.join(args.coverage_dir, 'samples.csv'), encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
  classes = list(dict.fromkeys(r['speed_class'] for r in sorted(rows, key=lambda r: r['speed_class'])))
  order = ['<= 50 km/h (urban)', '60-90 km/h (rural / arterial)', '100 km/h (rural highway / arterial)',
           '>= 110 km/h (motorway)']
  classes = [c for c in order if c in classes]
  towns = sorted({r['town'] for r in rows})
  per_cell = max(collections.Counter((r['town'], r['speed_class']) for r in rows).values())

  server = start_server(args.port)
  images = {}
  try:
    client = carla.Client('localhost', args.port)
    client.set_timeout(20.0)
    for _ in range(30):
      try:
        client.get_world()
        break
      except RuntimeError:
        time.sleep(5)
    client.set_timeout(600.0)
    for town in towns:
      print(f'loading {town}', flush=True)
      world = client.load_world(town)
      settings = world.get_settings()
      settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
      world.apply_settings(settings)
      world.set_weather(carla.WeatherParameters.ClearNoon)
      for r in [r for r in rows if r['town'] == town]:
        images.setdefault((town, r['speed_class']), []).append(
            snapshot(world, float(r['x']), float(r['y']), float(r['yaw'])))
      settings.synchronous_mode = False
      world.apply_settings(settings)
  finally:
    os.killpg(server.pid, 9)

  # Contact sheet: one row per town, one block of `per_cell` images per speed class.
  label_h, row_label_w = 40, 110
  sheet = Image.new('RGB', (row_label_w + len(classes) * per_cell * W, label_h + len(towns) * H), (30, 30, 30))
  draw = ImageDraw.Draw(sheet)
  for c, cls in enumerate(classes):
    draw.text((row_label_w + c * per_cell * W + 10, 12), cls, fill=(240, 240, 240))
  for t, town in enumerate(towns):
    draw.text((10, label_h + t * H + H // 2), town, fill=(240, 240, 240))
    for c, cls in enumerate(classes):
      for k, im in enumerate(images.get((town, cls), [])):
        sheet.paste(im, (row_label_w + (c * per_cell + k) * W, label_h + t * H))
  out = os.path.join(args.coverage_dir, 'snapshots.jpg')
  sheet.save(out, quality=85)
  print(f'wrote {out}', flush=True)


if __name__ == '__main__':
  main()
