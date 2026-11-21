export const revisionCohorts = {
  fresh: {title: 'Fresh matched trials', baseline: 'v0.1 rerun', source: 'same_study_rerun', pairs: 50},
  regression: {title: 'Known regression cases', baseline: 'Published v0.1', source: 'published_v0.1', pairs: 100}
};

const arms = ['baseline', 'candidate'];
const blockedCounts = {contacts: 'collision_events', yield_breaches: 'yield_violations', red_breaches: 'red_light_violations'};
const count = value => Number.isInteger(value) && value >= 0;

export function pairedCohort(report, stage) {
  const spec = revisionCohorts[stage];
  if (!spec || report?.stage !== stage || report.complete !== true || report.all_pairs_verified !== true ||
      report.completed_drives !== 100 || report.expected_drives !== 100 || report.matched_pairs !== spec.pairs ||
      report.baseline_source !== spec.source || !Array.isArray(report.setup_mismatches) || report.setup_mismatches.length ||
      !Array.isArray(report.cases) || report.cases.length !== 2 * spec.pairs) return null;

  const cases = new Map();
  const rows = {baseline: [], candidate: []};
  for (const row of report.cases) {
    const policy = stage === 'regression' && row?.policy === 'fusion' ? 'baseline' : row?.policy;
    if (!arms.includes(policy) || typeof row.case_id !== 'string' || !row.case_id || !Number.isInteger(row.seed) ||
        typeof row.route_id !== 'string' || !row.route_id || typeof row.outcome !== 'string' || !row.outcome ||
        typeof row.clean !== 'boolean' || !Object.values(blockedCounts).every(key => count(row[key]))) return null;
    const clean = row.outcome === 'complete' && !Object.values(blockedCounts).some(key => row[key]);
    if (clean !== row.clean) return null;
    if (!cases.has(row.case_id)) cases.set(row.case_id, new Map());
    const seeds = cases.get(row.case_id);
    if (!seeds.has(row.seed)) seeds.set(row.seed, {});
    if (seeds.get(row.seed)[policy]) return null;
    seeds.get(row.seed)[policy] = row;
    rows[policy].push(row);
  }

  const changes = {both_clean: 0, candidate_only: 0, baseline_only: 0, neither_clean: 0};
  for (const seeds of cases.values()) {
    for (const {baseline, candidate} of seeds.values()) {
      if (!baseline || !candidate || baseline.route_id !== candidate.route_id) return null;
      if (baseline.clean && candidate.clean) changes.both_clean++;
      else if (candidate.clean) changes.candidate_only++;
      else if (baseline.clean) changes.baseline_only++;
      else changes.neither_clean++;
    }
  }
  for (const arm of arms) {
    const totals = report.arms?.[arm];
    if (totals?.drives !== spec.pairs || totals.clean !== rows[arm].filter(row => row.clean).length ||
        Object.entries(blockedCounts).some(([key, field]) => totals[key] !== rows[arm].filter(row => row[field] > 0).length)) {
      return null;
    }
  }
  if (!Object.entries(changes).every(([key, value]) => (report.paired_clean?.[key] ?? 0) === value)) return null;
  return {spec, cases, totals: report.arms, changes};
}

export async function readRevisionReport(root, study, stage, fetchReport = fetch, cryptoApi = globalThis.crypto) {
  const digest = study.cohorts?.[stage]?.report_sha256;
  if (typeof digest !== 'string' || !/^[a-f0-9]{64}$/.test(digest) || !cryptoApi?.subtle) return null;
  const url = new URL(`${stage}.json`, root);
  const response = await fetchReport(url, {signal: AbortSignal.timeout(8000)});
  if (!response.ok) return null;
  const bytes = await response.arrayBuffer();
  const hash = await cryptoApi.subtle.digest('SHA-256', bytes);
  const actual = Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
  if (actual !== digest) return null;
  const report = JSON.parse(new TextDecoder().decode(bytes));
  return report.study_id === study.study_id ? pairedCohort(report, stage) : null;
}
