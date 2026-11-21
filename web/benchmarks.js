const policies = new Map([
  ['fusion', 'Learned sensor fusion'],
  ['camera', 'Learned camera-only'],
]);
const encounters = new Map([
  ['pedestrian', 'Pedestrian crossing'],
  ['pedestrian-delayed', 'Pedestrian crossing · delayed'],
  ['lead', 'Stopped lead vehicle'],
  ['braking', 'Braking lead vehicle'],
  ['occluded', 'Occluded pedestrian'],
  ['signal', 'Traffic signal'],
  ['left', 'Protected left turn'],
  ['turn', 'Unsignalized turn'],
  ['clear', 'Clear road'],
  ['trip', 'Destination trip'],
  ['trip-delayed', 'Destination trip · delayed'],
]);
const outcomes = new Map([
  ['complete', 'Clean finish'],
  ['collision', 'Collision'],
  ['red_light', 'Red-light violation'],
  ['timeout', 'Timeout'],
  ['lane_departure', 'Lane departure'],
  ['yield_violation', 'Yield violation'],
]);
const cohorts = new Map([
  ['primary', 'Primary evaluation'],
  ['camera_ablation', 'Camera-only subset'],
  ['supplemental', 'Supplemental trials'],
]);
const weatherLabels = new Map([
  ['ClearNoon', 'Clear noon'],
  ['CloudyNoon', 'Cloudy noon'],
  ['WetNoon', 'Wet noon preset'],
  ['ClearSunset', 'Clear sunset'],
]);
const state = document.getElementById('result-state');
const results = document.getElementById('results');
const encounterInput = document.getElementById('encounter');
const seedInput = document.getElementById('seed');

function element(tag, label) {
  const node = document.createElement(tag);
  if (label !== undefined) node.textContent = label;
  return node;
}

function tableHeader(label, scope) {
  const node = element('th', label);
  node.scope = scope;
  return node;
}

function humanize(value) {
  return value ? value.replace(/[_-]/g, ' ').replace(/^./, first => first.toUpperCase()) : '—';
}

function outcomeLabel(episode) {
  const value = episode?.outcome === 'complete' && episode.yield_violations > 0 ? 'yield_violation' : episode?.outcome;
  const label = element('span', value ? outcomes.get(value) ?? humanize(value) : 'No episode');
  label.className = 'outcome';
  if (value) label.dataset.value = value;
  return label;
}

function metric(name, value, unit, digits) {
  const item = element('div');
  const amount = Number.isFinite(value) ? [value.toFixed(digits), unit].filter(Boolean).join(' ') : '—';
  item.append(element('dt', name), element('dd', amount));
  return item;
}

function episodeCard(policy, episode) {
  const title = policies.get(policy);
  const card = element('article');
  card.className = 'episode-card';
  card.append(element('h3', title), outcomeLabel(episode));
  if (!episode) return card;

  const stats = element('dl');
  const { progress_m: progress, route_length_m: length } = episode;
  const hasRoute = Number.isFinite(progress) && Number.isFinite(length) && length > 0;
  const route = element('div');
  route.append(element('dt', 'Route progress'), element('dd', hasRoute ? `${progress.toFixed(1)} / ${length.toFixed(1)} m` : '—'));
  stats.append(route,
    metric('Simulated time', episode.duration_s, 's', 1),
    metric('Max lateral error', episode.max_lateral_error_m, 'm', 2));
  if (Number.isFinite(episode.collision_events)) stats.append(metric('Collision events', episode.collision_events, '', 0));
  if (Number.isFinite(episode.red_light_violations)) stats.append(metric('Red-light violations', episode.red_light_violations, '', 0));
  if (Number.isFinite(episode.yield_violations)) stats.append(metric('Yield violations', episode.yield_violations, '', 0));
  card.append(stats);

  if (hasRoute) {
    const bar = element('progress');
    bar.setAttribute('aria-label', `${title} route progress`);
    bar.max = length;
    bar.value = progress;
    card.append(bar);
  }
  return card;
}

function indexRecords(episodes) {
  const groups = new Map();
  for (const episode of episodes) {
    if (!episode || typeof episode.scenario !== 'string' || !episode.scenario || !Number.isInteger(episode.seed) ||
        typeof episode.policy !== 'string' || !episode.policy || typeof episode.outcome !== 'string' || !episode.outcome) {
      throw new Error('Unexpected episode summary');
    }
    const { case_id: caseId, scenario, seed, policy } = episode;
    const key = caseId || scenario;
    if (!groups.has(key)) groups.set(key, new Map());
    const seeds = groups.get(key);
    if (!seeds.has(seed)) seeds.set(seed, new Map());
    const paired = seeds.get(seed);
    if (paired.has(policy)) throw new Error('Duplicate episode summary');
    paired.set(policy, episode);
  }
  return groups;
}

function countOutcome(episodes, outcome) {
  return episodes.filter(episode => {
    if (outcome === 'collision') return episode.outcome === outcome || episode.collision_events > 0;
    if (outcome === 'red_light') return episode.outcome === outcome || episode.red_light_violations > 0;
    if (outcome === 'yield_violation') return episode.outcome === outcome || episode.yield_violations > 0;
    if (outcome === 'complete') {
      return episode.outcome === 'complete' &&
        !(episode.collision_events > 0 || episode.red_light_violations > 0 || episode.yield_violations > 0);
    }
    return episode.outcome === outcome;
  }).length;
}

function renderTotals(episodes, groups, policyKeys) {
  const totals = document.getElementById('policy-totals');
  const bins = new Map();
  for (const episode of episodes) {
    const cohort = cohorts.get(episode.cohort);
    const policy = policies.get(episode.policy);
    const label = cohort && episode.cohort !== 'camera_ablation' ? `${cohort} · ${policy}` : cohort || policy;
    if (!bins.has(label)) bins.set(label, []);
    bins.get(label).push(episode);
  }
  const rank = ([, records]) => {
    const cohort = [...cohorts.keys()].indexOf(records[0].cohort);
    return cohort >= 0 ? cohort : cohorts.size + policyKeys.indexOf(records[0].policy);
  };
  for (const [label, records] of [...bins].sort((a, b) => rank(a) - rank(b))) {
    const counts = [...outcomes.keys()].map(outcome => countOutcome(records, outcome));
    const row = element('tr');
    row.append(tableHeader(label, 'row'), ...[records.length, ...counts].map(count => element('td', count)));
    totals.append(row);
  }

  const pairs = [...groups.values()].flatMap(seeds => [...seeds.values()]);
  const matched = pairs.filter(pair => {
    const fusion = pair.get('fusion');
    const camera = pair.get('camera');
    return fusion && camera && (!camera.cohort || (camera.cohort === 'camera_ablation' && fusion.cohort === 'primary'));
  });
  const countLabel = (count, name) => `${count} ${name}${count === 1 ? '' : 's'}`;
  document.getElementById('coverage').textContent =
    `${countLabel(episodes.length, 'episode')} · ${countLabel(pairs.length, 'case/seed group')}`;
  if (matched.length) {
    const [fusion, camera] = ['fusion', 'camera'].map(policy => countOutcome(matched.map(pair => pair.get(policy)), 'complete'));
    const note = document.getElementById('matched-note');
    note.textContent = `Across ${countLabel(matched.length, 'matched trial')}, the multisensor policy completed ${fusion} of ${matched.length} cleanly and the camera-only policy completed ${camera} of ${matched.length} cleanly.`;
    note.hidden = false;
  }
}

function render(episodes, groups) {
  for (const {policy} of episodes) {
    if (!policies.has(policy)) policies.set(policy, humanize(policy));
  }
  const observedPolicies = new Set(episodes.map(episode => episode.policy));
  const policyKeys = [...policies.keys()].filter(policy => observedPolicies.has(policy));
  renderTotals(episodes, groups, policyKeys);
  document.getElementById('comparison').dataset.single = String(policyKeys.length === 1);
  document.getElementById('encounter-policies').append(...policyKeys.map(policy => tableHeader(policies.get(policy), 'col')));
  const cases = new Set([...encounters.keys(), ...groups.keys()]);
  for (const key of cases) {
    if (groups.has(key)) encounterInput.add(new Option(encounters.get(key) ?? humanize(key), key));
  }

  function showPair() {
    const seed = Number(seedInput.value);
    const seeds = groups.get(encounterInput.value);
    const pair = seeds.get(seed);
    const records = [...pair.values()];
    const weather = new Set(records.map(row => weatherLabels.get(row.weather) ?? row.weather).filter(Boolean));
    const turns = new Set(records.map(row => row.turn_direction).filter(Boolean));
    const context = [encounterInput.selectedOptions[0].textContent, `Seed ${seed}`];
    const destinations = new Set(records.map(row => row.destination).filter(Boolean));
    const delays = new Set(records.map(row => row.crossing_delay_s).filter(Number.isFinite));
    if (destinations.size === 1) context.push(humanize([...destinations][0]));
    if (delays.size === 1 && !delays.has(0)) context.push(`${[...delays][0]} s crossing delay`);
    if (turns.size === 1 && turns.has('left')) context.push('Left turn');
    if (turns.size === 1 && turns.has('right')) context.push('Right turn');
    if (weather.size === 1) context.push([...weather][0]);
    document.getElementById('comparison-context').textContent = context.join(' · ');
    document.getElementById('comparison').replaceChildren(...policyKeys.map(policy => episodeCard(policy, pair.get(policy))));

    const rows = [...seeds.keys()].sort((a, b) => a - b).map(candidate => {
      const row = element('tr');
      row.dataset.selected = String(candidate === seed);
      if (candidate === seed) row.setAttribute('aria-current', 'true');
      row.append(tableHeader(candidate, 'row'));
      for (const policy of policyKeys) {
        const cell = element('td');
        cell.append(outcomeLabel(seeds.get(candidate).get(policy)));
        row.append(cell);
      }
      return row;
    });
    document.getElementById('encounter-outcomes').replaceChildren(...rows);
  }

  function showEncounter() {
    const previous = seedInput.value;
    const seeds = [...groups.get(encounterInput.value).keys()].sort((a, b) => a - b);
    seedInput.replaceChildren(...seeds.map(seed => new Option(seed, seed)));
    if (seeds.includes(Number(previous))) seedInput.value = previous;
    showPair();
  }

  encounterInput.addEventListener('change', showEncounter);
  seedInput.addEventListener('change', showPair);
  showEncounter();
  state.hidden = true;
  results.hidden = false;
}

async function load() {
  try {
    const response = await fetch('./media/episodes.json');
    if (response.status === 404) {
      state.textContent = 'No evaluation episodes are available on this site yet.';
      return;
    }
    if (!response.ok) throw new Error('Episode summaries unavailable');
    const data = await response.json();
    const episodes = Array.isArray(data) ? data : data?.episodes;
    if (!Array.isArray(episodes)) throw new Error('Unexpected evaluation data');
    if (!episodes.length) {
      state.textContent = 'The evaluation file contains no episodes.';
      return;
    }
    render(episodes, indexRecords(episodes));
  } catch {
    state.textContent = 'Recorded evaluation episodes could not be loaded.';
  }
}

load();
