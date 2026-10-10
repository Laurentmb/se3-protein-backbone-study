"""Inspect file integrity and simple geometry; this does not test designability."""
import argparse
import csv
import json
from pathlib import Path
import warnings

from Bio.PDB import PDBParser
import numpy as np


def validate(path, expected_length):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        structure = PDBParser(QUIET=False).get_structure(path.stem,str(path))
    models = list(structure.get_models())
    assert len(models) == 1, 'Expected one PDB model'
    chains = list(models[0].get_chains())
    assert len(chains) == 1, 'Expected one chain'
    residues = list(chains[0].get_residues())
    assert len(residues) == expected_length
    for residue in residues:
        assert all(atom in residue for atom in ['N','CA','C','O']), 'Missing backbone atom'
    coords = np.array([[r[a].coord for a in ['N','CA','C','O']] for r in residues],dtype=float)
    assert np.isfinite(coords).all()
    numbers = [r.id[1] for r in residues]
    assert numbers == list(range(numbers[0],numbers[0]+expected_length)), 'Nonconsecutive numbering'
    ca = coords[:,1]
    ca_bonds = np.linalg.norm(np.diff(ca,axis=0),axis=-1)
    cn_bonds = np.linalg.norm(coords[:-1,2]-coords[1:,0],axis=-1)
    distances = np.linalg.norm(ca[:,None]-ca[None,:],axis=-1)
    upper = distances[np.triu_indices(len(ca),k=1)]
    all_coords = coords.reshape(-1,3)
    d = np.linalg.norm(all_coords[:,None]-all_coords[None,:],axis=-1)
    res_id = np.repeat(np.arange(len(residues)),4)
    valid_pairs = np.triu(np.abs(res_id[:,None]-res_id[None,:])>1,k=1)
    result = {
        'file':str(path.name), 'file_valid':True, 'models':len(models), 'chains':len(chains),
        'chain_id':chains[0].id, 'residues':len(residues), 'backbone_atoms':int(coords.shape[0]*4),
        'all_atoms':len(list(structure.get_atoms())), 'finite_coordinates':True,
        'residue_numbering_consecutive':True,
        'parser_warnings':[str(w.message) for w in caught],
        'ca_ca_mean_A':float(ca_bonds.mean()), 'ca_ca_min_A':float(ca_bonds.min()),
        'ca_ca_max_A':float(ca_bonds.max()),
        'ca_ca_mean_absolute_deviation_from_3_802_A':float(np.mean(np.abs(ca_bonds-3.80209737096))),
        'ca_breaks_over_4_5_A':int(np.count_nonzero(ca_bonds>4.5)),
        'ca_break_residue_pairs':[[numbers[i],numbers[i+1]] for i in np.flatnonzero(ca_bonds>4.5)],
        'ca_pairs_under_1_5_A':int(np.count_nonzero(upper<1.5)),
        'peptide_c_n_mean_A':float(cn_bonds.mean()), 'peptide_c_n_max_A':float(cn_bonds.max()),
        'peptide_c_n_pairs_over_2_A':int(np.count_nonzero(cn_bonds>2.0)),
        'nonadjacent_backbone_atom_pairs_under_1_5_A':int(np.count_nonzero((d<1.5)&valid_pairs)),
        'ca_radius_of_gyration_A':float(np.sqrt(np.mean(np.sum((ca-ca.mean(axis=0))**2,axis=-1)))),
    }
    metadata = json.loads((path.parent/'run_metadata.json').read_text(encoding='utf-8'))
    result.update({k:metadata[k] for k in ['model','num_steps','sampling_seconds','peak_gpu_allocated_mib','peak_gpu_reserved_mib']})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--length',type=int,default=100)
    args = parser.parse_args()
    rows = [validate(p,args.length) for p in sorted(args.root.glob('*/*.pdb'))]
    assert {r['model'] for r in rows} == {'framediff','frameflow'}
    (args.root/'validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    scalar_keys = [k for k,v in rows[0].items() if not isinstance(v,list)]
    with (args.root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=scalar_keys,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows,indent=2))
