"""Evaluate predicted folds against their source backbone, using CA atoms.

RMSD uses every residue in sequence order after a proper Kabsch alignment.
TM-align optimizes its own structural alignment and reports both normalizations.
Keep each candidate's metrics and the best-over-K result; never silently drop
failed predictions or substitute the source polyalanine for designed sequences.
"""
import argparse
import csv
import json
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
from tmtools import tm_align

from design_sequences import read_fasta


def read_structure(path):
    """Require one complete monomer and recover its sequence and CA coordinates."""
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    models = list(structure.get_models())
    assert len(models) == 1
    chains = list(models[0].get_chains())
    assert len(chains) == 1
    residues = list(chains[0].get_residues())
    assert all(all(a in residue for a in ('N', 'CA', 'C', 'O'))
        for residue in residues)
    all_positions = np.array([atom.coord for atom in structure.get_atoms()])
    assert np.isfinite(all_positions).all()
    numbering = [r.id[1] for r in residues]
    assert numbering == list(range(numbering[0], numbering[0]+len(residues)))
    ca = np.array([r['CA'].coord for r in residues], dtype=np.float64)
    sequence = ''.join(seq1(r.resname) for r in residues)
    confidence = np.array([r['CA'].bfactor for r in residues])
    return structure, ca, sequence, confidence


def kabsch(mobile, target):
    """Fit row-vector coordinates with a proper rotation, excluding reflections."""
    mobile_mean, target_mean = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile-mobile_mean).T @ (target-target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    offset = target_mean-mobile_mean @ rotation
    aligned = mobile @ rotation+offset
    distances = np.linalg.norm(aligned-target, axis=-1)
    return rotation, offset, distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--fasta', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, target, _, _ = read_structure(args.backbone)

    # A rigidly transformed control must produce effectively zero aligned RMSD.
    control_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    _, _, control = kabsch(target @ control_rotation+np.array([4., -2., 3.]), target)
    assert np.sqrt(np.mean(control**2)) < 1e-10
    rows = []
    for identifier, sequence in read_fasta(args.fasta):
        row = {'id': identifier, 'success': False}
        try:
            path = args.folds/(identifier+'.pdb')
            structure, mobile, recovered_sequence, confidence = read_structure(path)
            assert recovered_sequence == sequence, 'Predicted PDB sequence mismatch'
            assert len(mobile) == len(target) == len(sequence)
            rotation, offset, distances = kabsch(mobile, target)
            rmsd = float(np.sqrt(np.mean(distances**2)))
            # Cross-check Kabsch and RMSD with an independent BioPython routine.
            independent = SVDSuperimposer()
            independent.set(target, mobile)
            independent.run()
            assert abs(rmsd-independent.get_rms()) < 1e-8

            # Give both structures the candidate sequence; ALA labels in the
            # source PDB are placeholders, not biological sequence evidence.
            tm = tm_align(target, mobile, sequence, sequence)
            row.update(success=True, residues=len(mobile), ca_rmsd_A=rmsd,
                ca_mean_distance_A=float(distances.mean()),
                tm_score_normalized_backbone=float(tm.tm_norm_chain1),
                tm_score_normalized_prediction=float(tm.tm_norm_chain2),
                ca_mean_bfactor_confidence_raw=float(confidence.mean()),
                ca_min_bfactor_confidence_raw=float(confidence.min()),
                bfactor_confidence_scale=('0..1' if confidence.max() <= 1.0 else '0..100'),
                passes_rmsd_under_2A=bool(rmsd < 2.0))
            # Save a fitted PDB so the numerical comparison can be inspected.
            for atom in structure.get_atoms():
                atom.set_coord(atom.coord @ rotation+offset)
            writer = PDBIO()
            writer.set_structure(structure)
            writer.save(str(args.output/(identifier+'_aligned.pdb')))
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)

    successful = [r for r in rows if r['success']]
    summary = {'attempted_sequences': len(rows), 'successful_predictions': len(successful),
        'pipeline_verified': len(successful) == len(rows) and bool(rows),
        'rmsd_definition': 'sqrt(mean(squared CA distances)), all residues, Kabsch fit',
        'tm_score_definition': 'TM-align vended by tmtools 0.2.0, both normalizations',
        'metric_rigid_transform_control_rmsd_A': float(np.sqrt(np.mean(control**2))),
        'candidates': rows}
    if successful:
        best = min(successful, key=lambda r:r['ca_rmsd_A'])
        summary.update(best_sequence_by_rmsd=best['id'], scRMSD_A=best['ca_rmsd_A'],
            scTM=max(r['tm_score_normalized_backbone'] for r in successful),
            passes_scRMSD_under_2A=best['passes_rmsd_under_2A'])
    (args.output/'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'evaluation.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    if not summary['pipeline_verified']:
        raise RuntimeError('Pipeline verification is incomplete; see evaluation.json')
