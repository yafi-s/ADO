"""Temporarily score the released reader next to the current reader on one native drive."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter_ns

from crosswalk import signal_observer
from tools.rescore_signal_rgb import _digest, _metrics, _published_reader


_FIELDS = ('id', 'map_id', 'state', 'source', 'view', 'age_s', 'confidence', 'camera_support')


def _compact(readings):
    return [{key: row[key] for key in _FIELDS if key in row} for row in readings]


@contextmanager
def record_signal_comparison(destination):
    """Record original and current measurements before the scoring state is distributed."""
    original_module, original_sha = _published_reader()
    current = signal_observer.SignalObserver
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf8', newline='\n', buffering=1) as journal:
        provenance = dict(original_reader_sha256=original_sha,
                          revised_reader_sha256=_digest(Path(signal_observer.__file__)),
                          extractor_sha256=_digest(Path(__file__)),
                          classifier_sha256=_digest(signal_observer.CALIBRATION))
        journal.write(json.dumps({'producer': provenance}, separators=(',', ':')) + '\n')
        timing = dict(frames=0, selected_read_ns=0, baseline_read_ns=0, comparison_journal_ns=0)

        class PairedSignalObserver(current):
            def __init__(self, world):
                super().__init__(world)
                reference = object.__new__(original_module.SignalObserver)
                reference.classifier = original_module.HousingClassifier()
                reference.mounts = self.mounts
                reference.heads = self.heads
                reference.candidates = {}
                reference.recent = {}
                self._paired_world = world
                self._released_reader = reference
                self._sample = None

            def read(self, pose, images, sim_time=None):
                start = perf_counter_ns()
                revised = super().read(pose, images, sim_time)
                read_selected = perf_counter_ns()
                original = self._released_reader.read(pose, images, sim_time)
                read_baseline = perf_counter_ns()
                self._sample = dict(sim_time=sim_time, original=_compact(original), revised=_compact(revised))
                timing['selected_read_ns'] += read_selected - start
                timing['baseline_read_ns'] += read_baseline - read_selected
                timing['comparison_journal_ns'] += perf_counter_ns() - read_baseline
                return revised

            def score_reference(self, readings):
                states = super().score_reference(readings)
                start = perf_counter_ns()
                row = dict(frame=int(self._paired_world.get_snapshot().frame),
                           observer_map={'signal_reference': states}, **self._sample)
                journal.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                timing['comparison_journal_ns'] += perf_counter_ns() - start
                timing['frames'] += 1
                self._sample = None
                return states

        signal_observer.SignalObserver = PairedSignalObserver
        try:
            yield
        finally:
            signal_observer.SignalObserver = current
            journal.write(json.dumps({'timing': timing}, separators=(',', ':')) + '\n')


def _read_journal(path, arm, metadata):
    with path.open(encoding='utf8') as stream:
        metadata['producer'] = json.loads(stream.readline())['producer']
        first = last = timing = None
        for line in stream:
            if timing is not None or not line.endswith('\n'):
                raise ValueError('The paired reader journal has an incomplete or unexpected record')
            row = json.loads(line)
            if 'timing' in row:
                timing = row['timing']
                continue
            frame = row['frame']
            if last is not None and frame != last + 1:
                raise ValueError('The journal must contain one contiguous native drive')
            if first is None:
                first = frame
            last = frame
            yield dict(frame=frame, sim_time=row['sim_time'], observer_map=row['observer_map'], signals=row[arm])
    if first is None or timing is None:
        raise ValueError('The paired reader journal is missing frames or its closing record')
    count = last - first + 1
    if timing['frames'] != count:
        raise ValueError('The timing count must match the paired native frames')
    mean_timing_ms = {name.removesuffix('_ns'): ns / (1e6 * count) for name, ns in timing.items() if name.endswith('_ns')}
    metadata.update(first_frame=first, last_frame=last, frame_count=count, mean_wall_time_ms_per_frame=mean_timing_ms)


def report(path, out):
    evidence = dict(native_journal_sha256=_digest(path))
    evidence['comparisons'] = {arm: _metrics(_read_journal(path, arm, evidence), '2024')
                               for arm in ('original', 'revised')}
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n', encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('journal', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    report(args.journal, args.output)
