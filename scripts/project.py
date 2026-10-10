"""One entry point for offline analysis, tests and optional model inference.

Each model runs in its own process because upstream packages share names.
All paths are derived from this repository or supplied as explicit arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


def run(script, *arguments):
    """Use the active Python environment and propagate subprocess failures."""
    subprocess.run([sys.executable, '-B', str(SCRIPTS/script)] +
        [str(value) for value in arguments], check=True)


def writable_run(path):
    """Protect the bundled evidence from accidental output overwrites."""
    path = path.resolve()
    evidence = ROOT/'results'
    if path == ROOT or path == evidence or evidence in path.parents:
        raise SystemExit('Choose an output folder outside results/, for example runs/my-run.')
    return path


def verify():
    """Check every bundled result against its SHA-256 manifest."""
    manifest = json.loads((ROOT/'results/manifest.json').read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit('Result checksum mismatch: '+name)
    print('Verified {} bundled result files.'.format(len(manifest)), flush=True)


def analyze(source, output):
    """Recompute scores from raw PDBs and FASTA; make no external requests."""
    source, output = source.resolve(), writable_run(output)
    if source == output or source in output.parents:
        raise SystemExit('Analysis output must be separate from its input.')
    marker = output/'analysis_source.json'
    if output.exists() and any(output.iterdir()) and not marker.exists():
        raise SystemExit('Output contains other files. Choose an empty output folder.')
    identity = {'source': str(source)}
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise SystemExit('This output belongs to a different input experiment.')
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2))
    # Copy measured inputs; recomputed tables and fitted structures stay here.
    for name in ['protocol.json', 'generation.json', 'candidates.json',
                 'candidates.fasta', 'folding.json', 'numerical_reference.json']:
        if (source/name).exists():
            shutil.copyfile(str(source/name), str(output/name))
    for name in ['backbones', 'design', 'folds']:
        shutil.copytree(str(source/name), str(output/name), dirs_exist_ok=True)
    run('evaluate_pilot.py', '--output', output)
    run('plot_results.py', '--output', output)
    run('report_results.py', '--output', output)
    # Reproducing a known run must preserve every candidate score, not just averages.
    if (source/'sequence_results.json').exists():
        original = {r['id']: r for r in json.loads((source/'sequence_results.json').read_text())}
        reproduced = json.loads((output/'sequence_results.json').read_text())
        assert set(original) == {r['id'] for r in reproduced}
        for row in reproduced:
            assert row['success'] == original[row['id']]['success']
            if row['success']:
                for key in ['ca_rmsd_A', 'tm_score_normalized_backbone']:
                    assert abs(row[key]-original[row['id']][key]) < 1e-6, (row['id'], key)
        print('Every reproduced candidate score matches the saved experiment.', flush=True)
    print('Analysis saved to', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    analysis = commands.add_parser('analyze', help='Recompute the included pilot on CPU, offline')
    analysis.add_argument('--input', type=Path, default=ROOT/'results/pilot')
    analysis.add_argument('--output', type=Path, default=ROOT/'runs/reproduced-pilot')
    commands.add_parser('verify', help='Verify bundled result hashes')
    commands.add_parser('test-metrics', help='Test structural alignment and scoring')
    commands.add_parser('test-integrators', help='Run analytic CPU integrator tests; requires torch')
    commands.add_parser('prepare', help='Download pinned model sources and checkpoints')
    for name in ['generate', 'design', 'fold', 'numerical']:
        command = commands.add_parser(name)
        command.add_argument('--run', type=Path, required=True, help='Experiment folder, e.g. runs/new-pilot')
        if name in {'generate', 'numerical'}:
            command.add_argument('--repo', type=Path, default=ROOT/'external/frameflow')
            command.add_argument('--device', default='cuda:0')
        if name == 'generate':
            command.add_argument('--protocol', type=Path, default=ROOT/'configs/pilot.json')
            command.add_argument('--initial-states', type=Path, default=ROOT/'results/pilot/initial_states')
        if name == 'design':
            command.add_argument('--mpnn-repo', type=Path, default=ROOT/'external/framediff/ProteinMPNN')
            command.add_argument('--weights', type=Path, default=ROOT/'external/proteinmpnn_weights/v_48_020.pt')
        if name == 'fold':
            command.add_argument('--allow-sequence-export', action='store_true',
                help='Explicitly allow this run\'s candidate sequences to be sent to ESM Atlas')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
    elif args.command == 'analyze':
        verify() if args.input.resolve() == (ROOT/'results/pilot').resolve() else None
        analyze(args.input, args.output)
    elif args.command == 'test-metrics':
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
            '-s', str(ROOT/'tests'), '-v'], cwd=str(ROOT), check=True)
    elif args.command == 'test-integrators':
        run('test_integrators.py', '--output', ROOT/'runs/tests/integrator_tests.json')
    elif args.command == 'prepare':
        run('prepare_sources.py', '--output', ROOT/'external')
    else:
        folder = writable_run(args.run)
        if args.command == 'generate':
            if folder.exists() and any(folder.iterdir()):
                raise SystemExit('Choose an empty folder for a new generation experiment.')
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(args.protocol), str(folder/'protocol.json'))
            states = folder/'initial_states'
            states.mkdir()
            protocol = json.loads((folder/'protocol.json').read_text())
            for seed in protocol['seeds']:
                original = args.initial_states/('seed{}.pt'.format(seed))
                if original.exists():
                    shutil.copyfile(str(original), str(states/original.name))
            run('generate_pilot.py', '--repo', args.repo.resolve(), '--output', folder, '--device', args.device)
        elif args.command == 'design':
            run('design_pilot.py', '--mpnn-repo', args.mpnn_repo.resolve(), '--weights',
                args.weights.resolve(), '--output', folder)
        elif args.command == 'fold':
            if not args.allow_sequence_export:
                raise SystemExit('Folding sends amino-acid sequences to ESM Atlas. '
                    'Add --allow-sequence-export to authorize this batch.')
            run('fold_pilot.py', '--output', folder, '--allow-sequence-export')
        else:
            run('numerical_reference.py', '--repo', args.repo.resolve(), '--output', folder, '--device', args.device)


if __name__ == '__main__':
    main()
