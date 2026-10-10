"""Independent geometric controls for the metric and recorded experiment pairing."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_folds import kabsch


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        # An asymmetric non-coplanar cloud exposes reflection and RMSD mistakes.
        self.points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
            [0., 0., 3.], [1.2, -.3, .7]])

    def test_rigid_motion_is_removed(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        mobile = self.points @ rotation + np.array([7., -2., 1.])
        fitted, _, distances = kabsch(mobile, self.points)
        self.assertLess(np.sqrt(np.mean(distances**2)), 1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted), 1., places=12)

    def test_mirror_image_is_not_a_rigid_motion(self):
        reflected = self.points * np.array([-1., 1., 1.])
        rotation, _, distances = kabsch(reflected, self.points)
        self.assertAlmostEqual(np.linalg.det(rotation), 1., places=12)
        self.assertGreater(np.sqrt(np.mean(distances**2)), .1)

    def test_deformation_retains_nonzero_error(self):
        changed = self.points.copy()
        changed[-1] += [3., -2., 1.]
        _, _, distances = kabsch(changed, self.points)
        self.assertGreater(np.sqrt(np.mean(distances**2)), .1)


class ExperimentTests(unittest.TestCase):
    def test_pairs_share_input_and_budget(self):
        rows = json.loads((ROOT/'results/pilot/generation.json').read_text())
        pairs = {}
        for row in rows:
            pairs.setdefault((row['seed'], row['budget']), []).append(row)
        self.assertEqual(len(pairs), 10)
        for pair in pairs.values():
            self.assertEqual({r['method'] for r in pair}, {'euler', 'midpoint'})
            self.assertEqual(len({r['initial_state_sha256'] for r in pair}), 1)
            self.assertEqual(len({r['checkpoint_sha256'] for r in pair}), 1)
            for row in pair:
                factor = 1 if row['method'] == 'euler' else 2
                self.assertEqual(row['network_calls'], factor*row['intervals']+1)
                self.assertEqual(row['network_calls'], row['budget'])


if __name__ == '__main__':
    unittest.main()
