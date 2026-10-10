"""Plot C-alpha traces for inspection; does not predict secondary structure."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
args = parser.parse_args()
fig = plt.figure(figsize=(11,5),facecolor='#f5f7fa')
for i, name in enumerate(['framediff','frameflow']):
    path = next((args.root/name).glob('*.pdb'))
    ca = np.array([[float(line[j:j+8]) for j in [30,38,46]]
        for line in path.read_text().splitlines()
        if line.startswith('ATOM  ') and line[12:16].strip() == 'CA'])
    ca -= ca.mean(axis=0)
    ax = fig.add_subplot(1,2,i+1,projection='3d')
    ax.set_facecolor('#f5f7fa')
    segments = np.stack([ca[:-1],ca[1:]],axis=1)
    lines = Line3DCollection(segments,cmap='viridis',linewidths=2.7)
    lines.set_array(np.arange(len(segments)))
    ax.add_collection3d(lines)
    span = max(np.ptp(ca,axis=0).max()/2*1.15,10)
    ax.set_xlim(-span,span); ax.set_ylim(-span,span); ax.set_zlim(-span,span)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=20,azim=35)
    ax.scatter(*ca[[0,-1]].T,c=['#440154','#fde725'],s=35)
    ax.text(*ca[0],' N',fontsize=9)
    ax.text(*ca[-1],' C',fontsize=9)
    ax.set_axis_off()
    meta = json.loads((path.parent/'run_metadata.json').read_text())
    ax.set_title('{} | 100 residues\n{} steps | {:.1f} s sampling'.format(
        'FrameDiff' if name=='framediff' else 'FrameFlow',meta['num_steps'],meta['sampling_seconds']),
        fontsize=13,color='#12243a',pad=2)
fig.suptitle('First generated backbones',fontsize=18,fontweight='bold',color='#12243a',y=.98)
fig.text(.5,.035,'C-alpha traces, N to C: purple to yellow | Seed 123 | One sample per model',
    ha='center',fontsize=10,color='#526277')
fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.82,wspace=.02)
fig.savefig(str(args.root/'backbones_preview.png'),dpi=180,facecolor=fig.get_facecolor())
print(args.root/'backbones_preview.png')
