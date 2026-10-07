"""
Counterfactual tire-road friction intervention for CARLA 0.9.15.

Modes:
  none : nominal dry physics (no-op)
  ego  : scale the ego vehicle's per-wheel tire_friction (only the ego loses grip)
  road : spawn a town-sized static.trigger.friction volume (every vehicle inside loses grip)

The scale is relative to each vehicle's own default tire_friction, so 1.0 is always nominal.
"""
import carla

FRICTION_MODES = ('none', 'ego', 'road')


def apply_ego_friction(ego, scale):
  """Scales the ego's tire friction. Returns (base, requested) per-wheel lists for logging."""
  physics = ego.get_physics_control()
  wheels = physics.wheels
  base = [w.tire_friction for w in wheels]
  for w, b in zip(wheels, base):
    w.tire_friction = b * scale
  physics.wheels = wheels
  ego.apply_physics_control(physics)
  # Reading physics control back before the next tick can return stale values, so report the request.
  return base, [b * scale for b in base]


def apply_road_friction(world, friction_value):
  """
  Spawns one friction trigger covering the whole map. friction_value replaces each wheel's tire_friction
  for vehicles inside (verified: it acts on vehicles already inside when spawned, too).
  Centered at the origin on purpose: scenario_runner's ChangeRoadFriction places it at (-10000, -10000)
  and then it does not reach the town (no effect in calibration).
  """
  for actor in world.get_actors().filter('static.trigger.friction'):
    actor.destroy()
  bp = world.get_blueprint_library().find('static.trigger.friction')
  bp.set_attribute('friction', str(friction_value))
  for axis in ('x', 'y', 'z'):
    bp.set_attribute(f'extent_{axis}', str(1000000.0))
  return world.spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 0.0)))


def apply_friction(world, ego, mode, scale):
  """Entry point used by the leaderboard evaluator. Returns a dict describing what was applied."""
  if mode not in FRICTION_MODES:
    raise ValueError(f'Unknown friction mode {mode}, expected one of {FRICTION_MODES}')
  info = {'mode': mode, 'scale': scale}
  if mode == 'none' or scale == 1.0:
    return info
  if mode == 'ego':
    info['base_tire_friction'], info['applied_tire_friction'] = apply_ego_friction(ego, scale)
  elif mode == 'road':
    # Road trigger sets wheel friction directly, so express the target relative to the ego's default.
    base = ego.get_physics_control().wheels[0].tire_friction
    info['base_tire_friction'] = base
    info['trigger_friction'] = base * scale
    apply_road_friction(world, base * scale)
  return info


class FrictionEnforcer:
  """
  Applies the intervention and keeps the ego's tire friction at its target for the whole route.

  In the leaderboard, the ego's physics control is reset to the default during the first scenario ticks
  (measured: tire_friction 0.35 at tick 1, back to 3.5 at tick 2 and afterwards), so a one-shot
  apply_physics_control does not hold. enforce() is called every tick: it reads the wheels back and re-applies
  when they differ from the target. A read in the same tick as an apply returns stale values, so the check
  is skipped on the tick right after an apply. Every re-apply tick is recorded in info['reapply_ticks'].
  """

  def __init__(self, world, ego, mode, scale):
    self.ego = ego
    self.info = apply_friction(world, ego, mode, scale)
    self.info['reapply_ticks'] = []
    self.target = None
    if mode != 'none' and scale != 1.0:
      base = self.info['base_tire_friction']
      base = base if isinstance(base, list) else [base] * len(ego.get_physics_control().wheels)
      self.target = [b * scale for b in base]
    self._last_apply_tick = 0

  def enforce(self, tick):
    if self.target is None or tick <= self._last_apply_tick + 1:
      return
    physics = self.ego.get_physics_control()
    wheels = physics.wheels
    if all(abs(w.tire_friction - t) < 1e-3 for w, t in zip(wheels, self.target)):
      return
    for w, t in zip(wheels, self.target):
      w.tire_friction = t
    physics.wheels = wheels
    self.ego.apply_physics_control(physics)
    self._last_apply_tick = tick
    self.info['reapply_ticks'].append(tick)
