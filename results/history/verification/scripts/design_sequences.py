"""Design sequences for one generated backbone with the authors' ProteinMPNN.

The original polyalanine entry is excluded from candidates: only sampled
sequences are passed to the folding stage. All paths and settings are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read_fasta(path):
    """Read FASTA records, including sequences wrapped over several lines."""
    records = []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            records.append([line[1:], ''])
        elif line.strip():
            records[-1][1] += line.strip()
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=38)
    parser.add_argument('--num-sequences', type=int, default=2)
    args = parser.parse_args()
    repo, backbone, weights, output = [p.resolve() for p in
        (args.repo, args.backbone, args.weights, args.output)]
    output.mkdir(parents=True, exist_ok=True)

    # Explicit weights path avoids an upstream Unix path assumption on Windows.
    command = [sys.executable, str(repo / 'protein_mpnn_run.py'),
        '--pdb_path', backbone.as_posix(), '--pdb_path_chains', 'A',
        '--path_to_model_weights', str(weights.parent),
        '--model_name', weights.stem, '--out_folder', str(output),
        '--num_seq_per_target', str(args.num_sequences), '--batch_size', '1',
        '--sampling_temp', '0.1', '--seed', str(args.seed), '--device', '0']
    start = time.perf_counter()
    process = subprocess.run(command, cwd=str(repo), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    (output / 'proteinmpnn.log').write_text(process.stdout, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(process.stdout)

    # ProteinMPNN writes the input sequence first, followed by designed samples.
    records = read_fasta(output / 'seqs' / (backbone.stem + '.fa'))
    candidates = [(h, s) for h, s in records if 'sample=' in h]
    assert len(candidates) == args.num_sequences
    length = sum(line.startswith('ATOM  ') and line[12:16].strip() == 'CA'
        for line in backbone.read_text().splitlines())
    assert all(len(s) == length and set(s) <= set('ACDEFGHIKLMNPQRSTVWY')
        for _, s in candidates), 'Unexpected sequence length or alphabet'
    fasta = ''.join('>{}_seq{}\n{}\n'.format(backbone.stem, i, s)
        for i, (_, s) in enumerate(candidates, 1))
    (output / 'candidates.fasta').write_text(fasta, encoding='utf-8')
    metadata = {'backbone': str(backbone), 'model_name': weights.stem,
        'seed': args.seed, 'temperature': 0.1, 'num_sequences': len(candidates),
        'seconds': time.perf_counter() - start,
        'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=str(repo), text=True).strip(),
        'candidates': [{'id': '{}_seq{}'.format(backbone.stem, i),
            'sequence': s, 'upstream_header': h}
            for i, (h, s) in enumerate(candidates, 1)]}
    (output / 'design_metadata.json').write_text(json.dumps(metadata, indent=2),
        encoding='utf-8')
    print(json.dumps(metadata, indent=2))
