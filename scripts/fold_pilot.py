"""Refold the pilot's generated sequences through the common public ESM endpoint.

Reuse successful saved responses on restart, limit retries, and preserve failures.
The endpoint receives amino-acid sequences only. Calling this script requires
authorization to submit the candidate FASTA to the external folding service.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import urllib.error
import urllib.request


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-sequence-export',action='store_true')
    args=parser.parse_args()
    if not args.allow_sequence_export:
        parser.error('Add --allow-sequence-export to authorize sending this batch to ESM Atlas.')
    root=args.output.resolve()
    candidates=json.loads((root/'candidates.json').read_text())
    folder=root/'folds'
    folder.mkdir(exist_ok=True)
    endpoint='https://api.esmatlas.com/foldSequence/v1/pdb/'
    metadata_path=root/'folding.json'
    existing=json.loads(metadata_path.read_text()) if metadata_path.exists() else []
    rows={r['id']:r for r in existing}
    for index,candidate in enumerate(candidates):
        identifier=candidate['id']
        sequence_hash=hashlib.sha256(candidate['sequence'].encode()).hexdigest()
        # A saved success belongs to one exact sequence and one unmodified PDB.
        if identifier in rows and rows[identifier]['success']:
            saved=rows[identifier]
            if saved['sequence_sha256']!=sequence_hash:
                raise ValueError('Saved fold belongs to a different sequence: '+identifier)
            if hashlib.sha256((root/saved['pdb']).read_bytes()).hexdigest()!=saved['pdb_sha256']:
                raise ValueError('Saved prediction hash mismatch: '+identifier)
            continue
        row={'id':identifier,'backbone_id':candidate['backbone_id'],'success':False,
            'endpoint':endpoint,'sequence_length':len(candidate['sequence']),
            'sequence_sha256':sequence_hash,
            'requested_at_utc':datetime.now(timezone.utc).isoformat(),
            'attempts':rows.get(identifier,{}).get('attempts',[])}
        for attempt in range(2):
            status=None
            start=time.perf_counter()
            detail={'attempt':len(row['attempts'])+1,
                'requested_at_utc':datetime.now(timezone.utc).isoformat()}
            try:
                # Only the raw amino-acid sequence is transmitted in the POST body.
                request=urllib.request.Request(endpoint,data=candidate['sequence'].encode(),
                    headers={'Content-Type':'text/plain'},method='POST')
                with urllib.request.urlopen(request,timeout=40) as response:
                    status=response.status
                    content=response.read().decode('utf-8')
                    server_date=response.headers.get('Date')
                detail['http_status']=status
                if not any(line.startswith('ATOM  ') for line in content.splitlines()):
                    raise ValueError('Missing PDB atom records')
                path=folder/(identifier+'.pdb')
                path.write_text(content,encoding='utf-8')
                row.update(success=True,pdb=path.relative_to(root).as_posix(),
                    pdb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    server_date=server_date,
                    response_title=next((line for line in content.splitlines()
                        if line.startswith('TITLE')),''))
            except Exception as exc:
                detail['error']=str(exc)
                if isinstance(exc,urllib.error.HTTPError):
                    status=exc.code
                    detail['http_status']=status
                    detail['response_preview']=exc.read(300).decode('utf-8',errors='replace')
            detail['seconds']=time.perf_counter()-start
            row['attempts'].append(detail)
            if row['success']:
                break
            # Retry only temporary server/network failures; do not hammer 4xx.
            if status is not None and 400<=status<500:
                break
            time.sleep(2)
        rows[identifier]=row
        metadata_path.write_text(json.dumps(list(rows.values()),indent=2))
        print('Fold {}/{} {}: {}'.format(index+1,len(candidates),identifier,
            'OK' if row['success'] else 'FAILED'),flush=True)
    assert len(rows)==len(candidates)
