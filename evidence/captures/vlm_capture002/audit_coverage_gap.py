"""Reconcile raw GPU kernels against the exact filtered exporter input multiset."""
import gzip
import json
from collections import Counter
from pathlib import Path
import sys

from profiler_reports import _is_communication, _is_non_compute

run = Path(sys.argv[1])
with gzip.open(run / 'rank-0_report_events.json.gz', 'rt') as stream:
    snapshot = json.load(stream)
with gzip.open(run / 'reports/rank-0.json.gz', 'rt') as stream:
    events = json.load(stream)['traceEvents']
cpu_names = {e['name'] for e in events if e.get('cat') == 'cpu_op'}
counts = Counter()
for event in snapshot:
    if event['is_user_annotation']:
        continue
    for kernel in event['kernels']:
        if not _is_communication(event['key'], kernel['name']) and not _is_non_compute(event['key'], kernel['name'], cpu_names):
            counts[kernel['name'], round(kernel['duration'], 3)] += 1
missing = []
for event in events:
    if event.get('cat') != 'kernel' or _is_communication('', event['name']):
        continue
    key = event['name'], round(event['dur'], 3)
    if counts[key]:
        counts[key] -= 1
    else:
        missing.append(event)
ids = {e['args']['External id'] for e in missing if 'External id' in e['args']}
correlations = {e['args']['correlation'] for e in missing}
related = [e for e in events if e.get('args', {}).get('External id') in ids or e.get('args', {}).get('correlation') in correlations]
quality = json.loads((run / 'report_quality.json').read_text())
assert len(missing) == quality['trace_compute_kernel_events'] - quality['report_kernel_calls']
assert not sum(counts.values())
assert all(row['report_count'] > 0 for row in quality['kernel_name_count_mismatches'])
result = {
    'status': 'confirmed_count_gap',
    'method': 'Subtract the exact exporter-filtered snapshot (kernel name, duration rounded to 0.001 us) multiset from the raw compute-kernel multiset. Equal-duration occurrences are indistinguishable; listed occurrences are candidates, not independently proven event identities.',
    'missing_call_count': len(missing),
    'missing_kernel_names': [],
    'missing_time_us': sum(e['dur'] for e in missing),
    'unmatched_snapshot_calls_after_filtering': sum(counts.values()),
    'candidate_missing_trace_events': missing,
    'related_trace_events': related,
    'conclusion': 'The four-call deficit is already present in the filtered profiler.events() snapshot; all raw compute kernel names occur in the CSV. One candidate and its runtime launch have no External id; three candidates have cuDNN CPU associations in the trace. Exact upstream cause is not established. Reports and raw trace remain unmodified.',
}
(run / 'coverage_gap_investigation.json').write_text(json.dumps(result, indent=2) + '\n')
print({k: v for k, v in result.items() if k not in ('candidate_missing_trace_events', 'related_trace_events')})
