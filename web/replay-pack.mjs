import {applyBarrierReview,loadBarrierReview} from './barrier-review.mjs';

export function unpackReplay(pack) {
  const replay = pack?.replay;
  const frames = Array.isArray(replay) ? replay : replay?.frames;
  const path = pack?.shared_path;
  const ranges = pack?.path_ranges;
  if (pack?.format !== 'ado-replay-pack-v1' || !Array.isArray(frames) || !Array.isArray(ranges) ||
      (ranges.length ? !Array.isArray(path) : path !== null)) throw new Error('Invalid replay pack');

  let previousEnd = 0;
  for (const range of ranges) {
    if (!Array.isArray(range) || range.length !== 2) throw new Error('Invalid packed path range');
    const [begin, end] = range;
    if (!Number.isInteger(begin) || !Number.isInteger(end) || begin < previousEnd || end <= begin || end > frames.length) {
      throw new Error('Invalid packed path range');
    }
    for (let index = begin; index < end; index++) {
      const observer = frames[index]?.observer_map;
      if (!observer || typeof observer !== 'object' || Array.isArray(observer) || Object.hasOwn(observer, 'path')) {
        throw new Error('Invalid packed path reference');
      }
      observer.path = path;
    }
    previousEnd = end;
  }
  return replay;
}

async function fetchPacked(url, signal) {
  if (typeof url !== 'string' || typeof DecompressionStream !== 'function' || signal?.aborted) return null;
  const parsed = new URL(url, location.href);
  if (parsed.origin !== location.origin || !['https:', 'http:'].includes(parsed.protocol)) return null;
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, {once: true});
  const timer = setTimeout(cancel, 30000);
  try {
    const response = await fetch(parsed, {cache: 'no-cache', signal: controller.signal, redirect: 'error'});
    if (!response.ok || !response.body) return null;
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'), {signal: controller.signal});
    return unpackReplay(await new Response(stream).json());
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

export async function loadReplay(packUrl, signal, parse, raw, entry) {
  if (signal?.aborted) return null;
  const review=await loadBarrierReview(entry,signal);
  const parseReviewed=replay=>parse(applyBarrierReview(replay,review));
  try {
    const replay=await fetchPacked(packUrl,signal);
    if(replay!=null&&!signal?.aborted)return parseReviewed(replay);
  } catch {
    // Use the original replay if the optional pack cannot be fetched or read.
  }
  if (signal?.aborted) return null;
  const replay = await raw();
  return signal?.aborted ? null : parseReviewed(replay);
}
