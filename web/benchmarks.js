const policies = new Map([
  ['fusion', 'Learned sensor fusion'],
  ['camera', 'Learned camera-only'],
]);
const encounters = new Map([
  ['pedestrian', 'Pedestrian crossing'],
  ['lead', 'Stopped lead vehicle'],
  ['clear', 'Clear road'],
  ['turn', 'Unsignalized turn'],
  ['braking', 'Braking lead vehicle'],
]);
const outcomes = new Map([
  ['complete', 'Clean finish'],
  ['collision', 'Collision'],
  ['timeout', 'Timeout'],
  ['lane_departure', 'Lane departure'],
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

function outcomeLabel(episode) {
  const value = episode?.outcome;
  const label = element('span', outcomes.get(value) ?? 'No record');
  label.className = 'outcome';
  if (value) label.dataset.value = value;
  return label;
}

function metric(name, value, unit, digits) {
  const item = element('div');
  item.append(element('dt', name), element('dd', Number.isFinite(value) ? `${value.toFixed(digits)} ${unit}` : '—'));
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
    if (!episode || !encounters.has(episode.scenario) || !Number.isInteger(episode.seed) ||
        !policies.has(episode.policy) || !outcomes.has(episode.outcome)) {
      throw new Error('Unexpected episode summary');
    }
    const { scenario, seed, policy } = episode;
    if (!groups.has(scenario)) groups.set(scenario, new Map());
    const seeds = groups.get(scenario);
    if (!seeds.has(seed)) seeds.set(seed, new Map());
    const paired = seeds.get(seed);
    if (paired.has(policy)) throw new Error('Duplicate episode summary');
    paired.set(policy, episode);
  }
  return groups;
}

function render(episodes, groups) {
  const policyKeys = [...policies.keys()];
  const totals = document.getElementById('policy-totals');
  for (const [policy, title] of policies) {
    const records = episodes.filter(episode => episode.policy === policy);
    const counts = [...outcomes.keys()].map(outcome => records.filter(episode => episode.outcome === outcome).length);
    const row = element('tr');
    row.append(tableHeader(title, 'row'), ...[records.length, ...counts].map(count => element('td', count)));
    totals.append(row);
  }
  const groupCount = [...groups.values()].reduce((count, seeds) => count + seeds.size, 0);
  const countLabel = (count, name) => `${count} ${name}${count === 1 ? '' : 's'}`;
  document.getElementById('coverage').textContent =
    `${countLabel(episodes.length, 'episode')} · ${countLabel(groupCount, 'encounter/seed group')}`;
  document.getElementById('encounter-policies').append(...policyKeys.map(policy => tableHeader(policies.get(policy), 'col')));
  for (const [key, name] of encounters) {
    if (groups.has(key)) encounterInput.add(new Option(name, key));
  }

  function showPair() {
    const seed = Number(seedInput.value);
    const seeds = groups.get(encounterInput.value);
    const pair = seeds.get(seed);
    const records = [...pair.values()];
    const weather = new Set(records.map(row => weatherLabels.get(row.weather) ?? row.weather).filter(Boolean));
    const turns = new Set(records.map(row => row.turn_direction).filter(Boolean));
    const context = [encounters.get(encounterInput.value), `Seed ${seed}`];
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
    const episodes = await response.json();
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
