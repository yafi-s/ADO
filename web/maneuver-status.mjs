// The operator display follows the live planner's decision on this exposure.
const phases = {
  follow: ['following', 'Following traffic'],
  yield: ['waiting', 'Yielding before passing'],
  signal_left: ['passing', 'Signalling left'],
  change_left: ['passing', 'Moving into the passing lane'],
  wait_rejoin: ['waiting', 'Waiting to return'],
  signal_right: ['passing', 'Signalling right'],
  change_right: ['passing', 'Returning to the travel lane'],
  rejoined: ['moving', 'Pass complete'],
  route_merge: ['moving', 'Merging for the route']
};

export function maneuverActivity(pass) {
  if (pass?.planner !== 'route-maneuvers-v1') return null;
  const phase = pass.maneuver_phase;
  if (phase === 'passing' && pass.committed) {
    const label = pass.lead_kind === 'static' ? 'Passing a stationary object' : 'Passing';
    return {state: 'passing', label, planner: true};
  }
  if (!Object.hasOwn(phases, phase)) return null;
  const [state, label] = phases[phase];
  return {state, label, planner: true};
}
