"""
Exports a CARLA recorder log (from a run with RECORD=1) to an MP4 video.

The recording is replayed on a headless CARLA server in synchronous mode: every world tick advances the replay by
one recorded frame (0.05 s) and a camera attached to the replayed ego captures one video frame, so the video plays
in real time (20 fps) independent of rendering speed. Views: 'chase' (behind the ego), 'top' (bird's eye above the
ego) or 'both' (side by side). Each frame is labelled with the time, ego speed and an optional title.

Usage (no CARLA server needed, it starts one on --port; the GPU must be free):
  python friction_ood/scripts/replay_to_video.py <recording.log> --out run.mp4 [--view both] [--start 0] [--duration 20]
      [--title "TF++ route 14194, friction scale 0.1"] [--height 720] [--chase-pitch -6] [--top-height 25]
"""
import argparse
import os
import queue
import re
import socket
import subprocess
import time

import carla
import cv2
import numpy as np

GARAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DT = 0.05  # leaderboard recordings run at 20 Hz


def view_transforms(args):
  """Camera poses relative to the ego. A flatter, slightly higher chase camera shows more of the road ahead."""
  return {
      'chase': carla.Transform(carla.Location(x=-args.chase_back, z=args.chase_up), carla.Rotation(pitch=args.chase_pitch)),
      'top': carla.Transform(carla.Location(z=args.top_height), carla.Rotation(pitch=-90.0)),
  }


def start_server(port):
  env = dict(os.environ, VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
  server = subprocess.Popen([os.path.join(GARAGE, 'carla', 'CarlaUE4.sh'), '-RenderOffScreen', '-nosound',
                             f'-carla-rpc-port={port}'], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
  # A CARLA client call made before the server listens blocks for the whole timeout, so wait for the port first.
  for _ in range(120):
    with socket.socket() as s:
      if s.connect_ex(('localhost', port)) == 0:
        break
    time.sleep(2)
  return server


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('log', help='CARLA recorder .log file')
  parser.add_argument('--out', required=True, help='output .mp4')
  parser.add_argument('--view', choices=['chase', 'top', 'both'], default='both')
  parser.add_argument('--start', type=float, default=0.0, help='seconds into the recording')
  parser.add_argument('--duration', type=float, default=0.0, help='seconds to export (0 = until the end)')
  parser.add_argument('--title', default='', help='text drawn on every frame')
  parser.add_argument('--width', type=int, default=960, help='pixels per view')
  parser.add_argument('--height', type=int, default=720, help='pixels per view; taller shows more road ahead')
  parser.add_argument('--fov', type=float, default=90.0, help='horizontal field of view (deg)')
  parser.add_argument('--chase-back', type=float, default=8.0, help='chase camera distance behind the ego (m)')
  parser.add_argument('--chase-up', type=float, default=3.5, help='chase camera height (m)')
  parser.add_argument('--chase-pitch', type=float, default=-6.0, help='chase camera pitch (deg, negative = down)')
  parser.add_argument('--top-height', type=float, default=25.0, help='bird\'s-eye camera height (m)')
  parser.add_argument('--port', type=int, default=2000)
  args = parser.parse_args()
  log = os.path.abspath(args.log)

  server = start_server(args.port)
  try:
    client = carla.Client('localhost', args.port)
    client.set_timeout(20.0)
    for _ in range(30):
      try:
        client.get_world()
        break
      except RuntimeError:
        time.sleep(5)
    client.set_timeout(600.0)  # large maps take minutes to load

    info = client.show_recorder_file_info(log, False)
    town = re.search(r'Map: (\S+)', info).group(1)
    total = float(re.search(r'Duration: ([\d.]+)', info).group(1))
    duration = args.duration or (total - args.start)
    print(f'{os.path.basename(log)}: {town}, {total:.1f} s recorded, exporting {args.start:.1f}-{args.start + duration:.1f} s',
          flush=True)

    # Load the map ourselves so the replayer does not reload it, then switch to synchronous mode.
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, DT
    world.apply_settings(settings)
    client.replay_file(log, args.start, duration, 0)
    client.set_replayer_time_factor(1.0)
    for _ in range(5):
      world.tick()
    hero = next((a for a in world.get_actors().filter('vehicle.*') if a.attributes.get('role_name') == 'hero'), None)
    if hero is None:
      raise RuntimeError('No hero vehicle in the replay')

    names = ['chase', 'top'] if args.view == 'both' else [args.view]
    views = view_transforms(args)
    cams, frames = [], []
    for name in names:
      bp = world.get_blueprint_library().find('sensor.camera.rgb')
      for k, v in (('image_size_x', args.width), ('image_size_y', args.height), ('fov', args.fov)):
        bp.set_attribute(k, str(v))
      cam = world.spawn_actor(bp, views[name], attach_to=hero,
                              attachment_type=carla.AttachmentType.SpringArmGhost if name == 'chase'
                              else carla.AttachmentType.Rigid)
      q = queue.Queue()
      cam.listen(q.put)
      cams.append(cam)
      frames.append(q)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    size = (args.width * len(names), args.height)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), int(round(1 / DT)), size)
    n_frames = int(duration / DT)
    # The replayer moves actors kinematically, so get_velocity() is always zero: derive speed from the
    # displacement between frames instead, averaged over 0.25 s to smooth the recorded positions.
    prev_loc, recent = hero.get_location(), []
    for i in range(n_frames):
      world.tick()
      images = []
      for q in frames:
        image = q.get(timeout=30.0)
        images.append(np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3])
      frame = np.ascontiguousarray(np.hstack(images))
      if hero.is_alive:
        loc = hero.get_location()
        recent = (recent + [loc.distance(prev_loc) / DT])[-5:]
        prev_loc = loc
      speed = sum(recent) / len(recent) * 3.6 if recent else 0.0
      label = f't = {args.start + i * DT:5.2f} s   speed = {speed:5.1f} km/h'
      for text, y in ((args.title, 32), (label, 64)):
        if text:
          cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
          cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
      writer.write(frame)
      if i % 100 == 0:
        print(f'  frame {i}/{n_frames}', flush=True)
    writer.release()
    for cam in cams:
      cam.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print(f'wrote {args.out} ({n_frames} frames, {n_frames * DT:.1f} s)', flush=True)
  finally:
    os.killpg(server.pid, 9)


if __name__ == '__main__':
  main()
