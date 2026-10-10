"""Verify sequence export, bounded HTTP failures and cache reuse without networking."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]


class FoldingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.candidate = json.loads((ROOT/'results/pilot/candidates.json').read_text())[0]
        self.prediction = (ROOT/'results/pilot/folds'/(self.candidate['id']+'.pdb')).read_bytes()
        (self.folder/'candidates.json').write_text(json.dumps([self.candidate]))

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self):
        argv = ['fold_pilot.py', '--output', str(self.folder), '--allow-sequence-export']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT/'scripts/fold_pilot.py'), run_name='__main__')

    def test_saved_success_makes_no_request(self):
        (self.folder/'folds').mkdir()
        path = self.folder/'folds/prediction.pdb'
        path.write_bytes(self.prediction)
        row = {'id':self.candidate['id'], 'success':True, 'pdb':'folds/prediction.pdb',
            'sequence_sha256':hashlib.sha256(self.candidate['sequence'].encode()).hexdigest(),
            'pdb_sha256':hashlib.sha256(self.prediction).hexdigest()}
        (self.folder/'folding.json').write_text(json.dumps([row]))
        with patch('urllib.request.urlopen', side_effect=AssertionError('Unexpected export')) as request:
            self.invoke()
            request.assert_not_called()

    def test_post_body_contains_only_the_sequence(self):
        class Response:
            status = 200
            headers = {'Date':'recorded-test-date'}
            def __enter__(response):
                return response
            def __exit__(response, *args):
                return False
            def read(response):
                return self.prediction
        with patch('urllib.request.urlopen', return_value=Response()) as call:
            self.invoke()
        request = call.call_args[0][0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(request.data, self.candidate['sequence'].encode())
        row = json.loads((self.folder/'folding.json').read_text())[0]
        self.assertTrue(row['success'])

    def test_client_error_is_saved_without_retry(self):
        error = urllib.error.HTTPError('https://example.invalid/', 400, 'Bad Request', {}, io.BytesIO(b'test error'))
        with patch('urllib.request.urlopen', side_effect=error) as call:
            self.invoke()
        self.assertEqual(call.call_count, 1)
        row = json.loads((self.folder/'folding.json').read_text())[0]
        self.assertFalse(row['success'])
        self.assertEqual(row['attempts'][0]['http_status'], 400)


if __name__ == '__main__':
    unittest.main()
