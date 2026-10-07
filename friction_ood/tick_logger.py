"""
Per-tick log of ego state, applied control and the agent's intended plan, for plan-vs-reality divergence.

One gzipped JSON line per simulator tick. The ego state is the state the agent observed when choosing the control
in the same row (it is read before the world is ticked again). Vehicle-frame quantities use CARLA's convention:
x forward, y right, z up. The plan is copied from agent.last_plan and only written when it changes (its 'step'
differs from the last logged one), so rows without 'plan' reuse the previous plan.
"""
import gzip
import json
import math


def _dot(a, b):
  return a.x * b.x + a.y * b.y + a.z * b.z


class TickLogger:

  def __init__(self, path, meta=None):
    self.path = path
    self._file = gzip.open(path, 'wt', encoding='utf-8')
    self._last_plan_step = None
    if meta:
      self._file.write(json.dumps({'meta': meta}) + '\n')

  def log(self, tick, game_time, ego, control, plan=None):
    tf = ego.get_transform()
    fwd, right = tf.get_forward_vector(), tf.get_right_vector()
    vel, acc, ang = ego.get_velocity(), ego.get_acceleration(), ego.get_angular_velocity()
    row = {
        'tick': tick,
        't': game_time,
        'x': tf.location.x,
        'y': tf.location.y,
        'yaw': tf.rotation.yaw,  # degrees
        'vx': _dot(vel, fwd),
        'vy': _dot(vel, right),
        'ax': _dot(acc, fwd),
        'ay': _dot(acc, right),
        'yaw_rate': math.radians(ang.z),  # CARLA reports deg/s
        'steer': control.steer,
        'throttle': control.throttle,
        'brake': control.brake,
    }
    if tick in (1, 2, 20) or tick % 200 == 0:
      # Ground truth that the friction intervention is in effect (a same-tick read after applying is stale).
      row['tire_friction'] = [w.tire_friction for w in ego.get_physics_control().wheels]
    if plan is not None and plan.get('step') != self._last_plan_step:
      row['plan'] = plan
      self._last_plan_step = plan.get('step')
    self._file.write(json.dumps(row) + '\n')

  def close(self):
    if self._file is not None:
      self._file.close()
      self._file = None
