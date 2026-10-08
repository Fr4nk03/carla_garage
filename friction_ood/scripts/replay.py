"""
Replays a CARLA recorder log (from a run with RECORD_PATH set) with the spectator camera following the ego.

The replay is kinematic playback of the recorded actor states, so it shows exactly what happened in that run,
including slip under the friction intervention (physics is not re-simulated).

Usage (a CARLA server must be running with a window, see replay.sh):
  python friction_ood/scripts/replay.py <recording.log> [--speed 0.5] [--start 0] [--duration 0] [--view chase|top]
"""
import argparse
import re
import time

import carla


def find_hero_id(client, log_path):
  """The recorder info lists every spawned actor; the ego is the vehicle with role_name = hero."""
  info = client.show_recorder_file_info(log_path, True)
  actor_id = None
  for line in info.splitlines():
    created = re.search(r'Create (\d+): (\S+)', line)
    if created:
      actor_id = int(created.group(1)) if created.group(2).startswith('vehicle.') else None
    elif actor_id is not None and re.search(r'role_name\s*=\s*hero', line):
      return actor_id, info
  raise RuntimeError('No hero vehicle found in the recording')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('log', help='absolute path to the .log written by the CARLA recorder')
  parser.add_argument('--host', default='localhost')
  parser.add_argument('--port', type=int, default=2000)
  parser.add_argument('--speed', type=float, default=1.0, help='replay time factor (0.5 = half speed)')
  parser.add_argument('--start', type=float, default=0.0, help='start time in seconds')
  parser.add_argument('--duration', type=float, default=0.0, help='seconds to replay (0 = all)')
  parser.add_argument('--view', choices=['chase', 'top'], default='chase')
  args = parser.parse_args()

  client = carla.Client(args.host, args.port)
  client.set_timeout(300.0)
  hero_id, info = find_hero_id(client, args.log)
  total = float(re.search(r'Duration: ([\d.]+)', info).group(1))
  print(f'Recording duration {total:.1f} s, ego actor id {hero_id}')

  # Chase view: the replayer moves the spectator behind the followed actor every frame.
  print(client.replay_file(args.log, args.start, args.duration, hero_id if args.view == 'chase' else 0))
  client.set_replayer_time_factor(args.speed)

  # The replayer loads the recorded map first (minutes for large maps); start timing once it is playing.
  town = re.search(r'Map: (\S+)', info).group(1)
  world = client.get_world()
  while not world.get_map().name.endswith(town):
    time.sleep(2.0)
    world = client.get_world()
  # Replayed actors get new ids; find the ego by its role to drive the top view.
  hero = None
  for _ in range(30):
    hero = next((a for a in world.get_actors().filter('vehicle.*') if a.attributes.get('role_name') == 'hero'), None)
    if hero is not None:
      break
    world.wait_for_tick(10.0)
  length = (args.duration or (total - args.start)) / args.speed
  end = time.time() + length + 5.0
  while time.time() < end:
    if args.view == 'top' and hero is not None and hero.is_alive:
        loc = hero.get_location()
        world.get_spectator().set_transform(carla.Transform(loc + carla.Location(z=50), carla.Rotation(pitch=-90)))
    world.wait_for_tick(10.0)
  print('Replay finished. Re-run this script to watch again (the CARLA window stays open).')


if __name__ == '__main__':
  main()
