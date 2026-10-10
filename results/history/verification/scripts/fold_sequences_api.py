"""Fold explicitly authorized candidate sequences with the documented ESM API.

Only amino-acid sequences are submitted. Save raw PDB responses and timings;
record failures instead of treating them as successful structure predictions.
The public service does not expose a verifiable checkpoint or recycle setting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import requests

from design_sequences import read_fasta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fasta', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
    rows = []
    for identifier, sequence in read_fasta(args.fasta):
        row = {'id': identifier, 'sequence_length': len(sequence),
            'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'endpoint': endpoint, 'success': False}
        start = time.perf_counter()
        response = None
        try:
            response = requests.post(endpoint, data=sequence,
                headers={'Content-Type': 'text/plain'}, timeout=40)
            row['http_status'] = response.status_code
            response.raise_for_status()
            if not any(line.startswith('ATOM  ') for line in response.text.splitlines()):
                raise ValueError('Response does not contain PDB atom records')
            path = args.output / (identifier+'.pdb')
            path.write_text(response.text, encoding='utf-8')
            row.update(success=True, pdb=path.name,
                pdb_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as exc:
            row['error'] = str(exc)
            if response is not None:
                row['response_preview'] = response.text[:500]
        row['seconds'] = time.perf_counter()-start
        rows.append(row)
        (args.output/'fold_metadata.json').write_text(json.dumps(rows, indent=2),
            encoding='utf-8')
        print(json.dumps(row, indent=2), flush=True)
