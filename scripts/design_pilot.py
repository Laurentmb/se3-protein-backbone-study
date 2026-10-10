"""Design the same number of sequences for every pilot backbone, without folding.

Each backbone restarts ProteinMPNN with the same seed and temperature.
Save one combined FASTA plus a mapping from every sequence to its backbone.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpnn-repo',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    runs=json.loads((output/'generation.json').read_text())
    candidates=[]
    for index, run in enumerate(runs):
        folder=output/'design'/run['id']
        command=[sys.executable,'-B',str(Path(__file__).parent/'design_sequences.py'),
            '--repo',str(args.mpnn_repo.resolve()),
            '--backbone',str(output/run['pdb']),
            '--weights',str(args.weights.resolve()),
            '--output',str(folder),'--seed',str(protocol['proteinmpnn']['seed_per_backbone']),
            '--num-sequences',str(protocol['proteinmpnn']['sequences_per_backbone']),
            '--temperature',str(protocol['proteinmpnn']['temperature'])]
        if not (folder/'design_metadata.json').exists():
            process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if process.returncode:
                raise RuntimeError(process.stdout)
        metadata=json.loads((folder/'design_metadata.json').read_text())
        for candidate in metadata['candidates']:
            candidates.append({**candidate,'backbone_id':run['id'],
                'seed':run['seed'],'budget':run['budget'],'method':run['method']})
        (output/'candidates.json').write_text(json.dumps(candidates,indent=2))
        (output/'candidates.fasta').write_text(''.join('>{}\n{}\n'.format(c['id'],c['sequence'])
            for c in candidates),encoding='utf-8')
        print('Designed {}/{} backbones'.format(index+1,len(runs)),flush=True)
    assert len(candidates)==len(runs)*protocol['proteinmpnn']['sequences_per_backbone']
