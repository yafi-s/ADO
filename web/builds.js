import {artifactUrl, previewImage, setArtifactLink, validRecordingPath} from './build-data.js';

const byId = id => document.getElementById(id);
const decimal = new Intl.NumberFormat(undefined, {maximumFractionDigits: 1});
const policyLabels = {fusion: 'Learned multisensor', camera: 'Learned vision', teacher: 'Training instructor'};
const outcomeLabels = {complete: 'Route complete', red_light: 'Red-light violation'};

function words(value) {
  if (!value) return '—';
  const words = value.replace(/[_-]/g, ' ');
  return words[0].toUpperCase() + words.slice(1);
}

function link(label, url) {
  const element = document.createElement('a');
  element.textContent = label;
  element.href = url;
  return element;
}

function detail(section, id, value) {
  byId(section).hidden = !value;
  byId(id).textContent = value ?? '';
}

function renderBuild(build) {
  byId('build-name').textContent = build.release?.label || build.title || 'CARLA simulation build';
  const statusLabels = {commissioning: 'Commissioning', evaluated: 'Evaluation archived'};
  byId('build-status').textContent = statusLabels[build.status] || '';
  const commit = build.source?.commit;
  const commitLink = byId('source-commit-link');
  const commitText = byId('source-commit-text');
  byId('source-info').hidden = !commit;
  commitLink.textContent = commit || '';
  commitLink.className = 'source-commit';
  setArtifactLink(commitLink, build.source?.url);
  commitText.hidden = !commitLink.hidden;
  commitText.textContent = commit || '';
  detail('server-info', 'server-version', build.simulator?.server_version);
  detail('client-info', 'client-version', build.simulator?.client_version);
  detail('map-info', 'simulator-map', build.simulator?.map);
  setArtifactLink(byId('release-zip'), build.release?.zip_url);
  setArtifactLink(byId('release-link'), build.release?.url);
  setArtifactLink(byId('release-checksum'), build.release?.sha256_url);
  byId('release-note').hidden = !byId('release-zip').hidden;
}

function renderEvidence(build) {
  const items = [];
  const seen = new Set();
  function add(label, url) {
    if (!url || !label || seen.has(url)) return;
    const item = document.createElement('li');
    item.append(link(label, url));
    items.push(item);
    seen.add(url);
  }
  const video = artifactUrl(build.preview_video);
  const image = previewImage(build);
  if (video || image) add(video ? 'Watch CARLA training demonstration' : 'View CARLA camera capture', artifactUrl('./'));
  if (video) add('CARLA training demonstration MP4', video);
  if (image) add('CARLA camera capture', image);
  for (const item of Array.isArray(build.evidence) ? build.evidence : []) {
    add(item?.label, artifactUrl(item?.url));
  }
  byId('evidence-list').replaceChildren(...items);
  byId('evidence-panel').hidden = !items.length;
}

function renderCheckpoints(checkpoints) {
  const rows = [];
  for (const [policy, checkpoint] of Object.entries(checkpoints ?? {})) {
    if (!checkpoint?.sha256) continue;
    const row = document.createElement('div');
    const label = document.createElement('dt');
    label.textContent = policyLabels[policy] || words(policy);
    const data = document.createElement('dd');
    const hash = document.createElement('code');
    hash.textContent = checkpoint.sha256;
    hash.setAttribute('aria-label', 'SHA-256 ' + checkpoint.sha256);
    data.append(hash);
    const url = artifactUrl(checkpoint.url);
    if (url) data.append(link('Download checkpoint', url));
    row.append(label, data);
    rows.push(row);
  }
  byId('checkpoints').replaceChildren(...rows);
  byId('checkpoint-panel').hidden = !rows.length;
}

function assignOptions(select, title, values, labels) {
  select.replaceChildren(new Option(title, ''), ...[...new Set(values.filter(Boolean))]
    .sort()
    .map(value => new Option(labels[value] ?? words(value), value)));
}

function renderTrial(run) {
  const row = document.createElement('tr');
  const title = document.createElement('th');
  title.scope = 'row';
  title.textContent = run.trial_id;
  const caseLabel = document.createElement('span');
  caseLabel.className = 'trial-case';
  caseLabel.textContent = [words(run.case_id), run.cohort && words(run.cohort)].filter(Boolean).join(' · ');
  title.append(caseLabel);
  row.append(title);
  row.insertCell().textContent = run.seed ?? '—';
  row.insertCell().textContent = policyLabels[run.policy] ?? words(run.policy);
  const outcome = document.createElement('span');
  outcome.textContent = outcomeLabels[run.outcome] ?? words(run.outcome);
  outcome.className = 'trial-outcome';
  if (run.outcome) outcome.dataset.outcome = run.outcome;
  row.insertCell().append(outcome);
  row.insertCell().textContent = Number.isFinite(run.progress_m) ? decimal.format(run.progress_m) + ' m' : '—';
  row.insertCell().textContent = Number.isFinite(run.duration_s) ? decimal.format(run.duration_s) + ' s' : '—';
  const evidence = document.createElement('div');
  evidence.className = 'trial-evidence';
  if (validRecordingPath(run.recording_path)) {
    const url = new URL('./', document.baseURI);
    url.searchParams.set('run', run.trial_id);
    evidence.append(link('Watch replay', url.href));
  }
  const log = artifactUrl(run.log_url);
  if (log) evidence.append(link('Run log', log));
  row.insertCell().append(evidence.childElementCount ? evidence : '—');
  return row;
}

function renderRuns(build) {
  const runs = Array.isArray(build.runs) ? build.runs.filter(run => run?.trial_id) : [];
  setArtifactLink(byId('runs-csv'), build.runs_csv);
  byId('trials').hidden = !runs.length;
  byId('trials-empty').hidden = Boolean(runs.length);
  if (!runs.length) return;

  const search = byId('trial-search');
  const policy = byId('trial-policy');
  const outcome = byId('trial-outcome');
  assignOptions(policy, 'All controllers', runs.map(run => run.policy), policyLabels);
  assignOptions(outcome, 'All outcomes', runs.map(run => run.outcome), outcomeLabels);
  const rows = runs.map(run => ({
    row: renderTrial(run),
    run,
    searchText: [run.trial_id, run.case_id, words(run.case_id), run.seed, run.cohort].join(' ').toLowerCase()
  }));
  byId('trial-rows').replaceChildren(...rows.map(({row}) => row));
  function filter() {
    const terms = search.value.trim().toLowerCase().split(/\s+/);
    let matches = 0;
    for (const {row, run, searchText} of rows) {
      const visible = (!policy.value || run.policy === policy.value) &&
        (!outcome.value || run.outcome === outcome.value) && terms.every(term => searchText.includes(term));
      row.hidden = !visible;
      if (visible) matches++;
    }
    byId('trial-count').textContent = matches ? matches + ' of ' + runs.length + ' runs' : 'No matching runs';
  }
  search.addEventListener('input', filter);
  policy.addEventListener('change', filter);
  outcome.addEventListener('change', filter);
  filter();
}

async function showArchive() {
  const state = byId('archive-state');
  try {
    const response = await fetch('./media/build.json', {cache: 'no-store', signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error('Archive request failed');
    const build = await response.json();
    if (!build || typeof build !== 'object' || Array.isArray(build)) throw new Error('Invalid archive');
    renderBuild(build);
    renderEvidence(build);
    renderCheckpoints(build.checkpoints);
    renderRuns(build);
    byId('build-archive').hidden = false;
    state.hidden = true;
  } catch {
    state.textContent = 'The build archive is temporarily unavailable.';
  }
}

showArchive();
