import {readRevisionReport, revisionCohorts} from './benchmarks-v2.mjs';

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

function resultsTable(label, columns) {
  const wrap = element('div');
  wrap.className = 'table-wrap';
  wrap.setAttribute('role', 'region');
  wrap.setAttribute('aria-label', label);
  wrap.tabIndex = 0;
  const table = element('table'), head = element('thead'), row = element('tr'), body = element('tbody');
  row.append(...columns.map(column => tableHeader(column, 'col')));
  head.append(row);
  table.append(head, body);
  wrap.append(table);
  return {wrap, body};
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

function revisionPanel(stage, data, root) {
  const {spec, cases, totals, changes} = data;
  const panel = element('section'), heading = element('h3', spec.title), header = element('header');
  panel.className = 'panel';
  heading.id = `revision-${stage}`;
  panel.setAttribute('aria-labelledby', heading.id);
  header.className = 'panel-head result-head';
  const paired = element('span', `${spec.pairs} matched pairs`);
  paired.className = 'unit-tag';
  header.append(heading, paired);

  const totalsTable = resultsTable(`${spec.title} totals`, ['Policy', 'Clean finishes', 'Collisions', 'Yield violations', 'Red-light violations']);
  for (const [arm, label] of [['baseline', spec.baseline], ['candidate', 'Behavior V2']]) {
    const scores = totals[arm];
    const row = element('tr');
    row.append(tableHeader(label, 'row'), element('td', `${scores.clean} / ${scores.drives}`),
      ...[scores.contacts, scores.yield_breaches, scores.red_breaches].map(value => element('td', value)));
    totalsTable.body.append(row);
  }
  const context = stage === 'fresh' ? 'Both policies ran the same newly selected case and seed pairs.' :
    'Behavior V2 repeated the routes and seeds used in the published v0.1 evaluation.';
  const note = element('p', `${context} A clean finish reached the route end without contact or traffic violations. ` +
    'Collision and violation totals count affected runs. ' +
    `Both finished cleanly in ${changes.both_clean} pairs. Behavior V2 alone finished cleanly in ${changes.candidate_only}, ` +
    `v0.1 alone in ${changes.baseline_only}, and neither in ${changes.neither_clean}. `);
  note.className = 'result-note';
  const link = element('a', 'View cohort report');
  link.href = new URL(`${stage}.json`, root).href;
  note.append(link);

  const details = element('details'), picker = element('div'), select = element('select');
  details.className = 'revision-pairs';
  picker.className = 'source-picker';
  select.id = `revision-case-${stage}`;
  const label = element('label', 'Case');
  label.htmlFor = select.id;
  picker.append(label, select);
  const pairs = resultsTable(`${spec.title} outcomes by seed`, ['Seed', spec.baseline, 'Behavior V2']);
  const options = [...new Set([...encounters.keys(), ...cases.keys()])].filter(key => cases.has(key));
  select.replaceChildren(...options.map(key => new Option(key === 'left' ? 'Left turn' : encounters.get(key) ?? humanize(key), key)));
  function showCase() {
    const selected = cases.get(select.value);
    const rows = [...selected].sort((a, b) => a[0] - b[0]).map(([seed, {baseline, candidate}]) => {
      const row = element('tr');
      row.append(tableHeader(seed, 'row'));
      for (const episode of [baseline, candidate]) {
        const cell = element('td');
        cell.append(outcomeLabel(episode));
        row.append(cell);
      }
      return row;
    });
    pairs.body.replaceChildren(...rows);
  }
  select.addEventListener('change', showCase);
  showCase();
  details.append(element('summary', 'Matched outcomes by case and seed'), picker, pairs.wrap);
  panel.append(header, totalsTable.wrap, note, details);
  return panel;
}

async function loadRevision() {
  const roots = [new URL('./media/behavior-v2/', location.href),
    new URL('https://raw.githubusercontent.com/yafi-s/ADO/main/results/behavior-v2/')];
  if (['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) {
    roots.splice(1, 0, new URL('../results/behavior-v2/', location.href));
  }
  for (const root of roots) {
    const manifestUrl = new URL('study.json', root);
    try {
      const response = await fetch(manifestUrl, {signal: AbortSignal.timeout(8000)});
      if (!response.ok) continue;
      const study = await response.json();
      if (typeof study?.study_id !== 'string' || !study.study_id || study.archive?.file !== 'closed-loop-evidence.zip' ||
          typeof study.archive.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(study.archive.sha256)) continue;
      const reports = await Promise.allSettled(Object.keys(revisionCohorts).map(async stage => {
        const report = await readRevisionReport(root, study, stage);
        return report && revisionPanel(stage, report, root);
      }));
      const panels = reports.filter(result => result.status === 'fulfilled' && result.value).map(result => result.value);
      if (!panels.length) continue;
      document.getElementById('revision-cohorts').replaceChildren(...panels);
      document.getElementById('revision-download').href = new URL(study.archive.file, root).href;
      document.getElementById('revision-manifest').href = manifestUrl.href;
      document.getElementById('revision-results').hidden = false;
      document.getElementById('legacy-title').hidden = false;
      return;
    } catch {
      // Sites without the new evidence still show the original published evaluation.
    }
  }
}

load();
void loadRevision();

async function loadCityReview() {
  const response = await fetch('./media/urban-review.json', {cache: 'no-cache'});
  if (!response.ok) return;
  const study = await response.json();
  if (!Array.isArray(study.checks) || !study.checks.length) return;
  const rows = [];
  for (const check of study.checks) {
    const row = document.createElement('tr');
    const title = element('th', check.test);
    title.scope = 'row';
    const links = document.createElement('td');
    for (const entry of check.evidence) {
      const url = new URL(entry.url, document.baseURI);
      if (url.origin !== location.origin && url.protocol !== 'https:') continue;
      if (links.childNodes.length) links.append(' · ');
      const link = element('a', entry.label);
      link.href = url.href;
      links.append(link);
    }
    row.append(title, element('td', check.result), links);
    rows.push(row);
  }
  document.getElementById('city-review-rows').replaceChildren(...rows);
  document.getElementById('city-review-limits').textContent = study.scope;
  document.getElementById('city-review').hidden = false;
}
void loadCityReview().catch(() => {});
