"""Write a short English results report from computed scores, without torch."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    summary = json.loads((root/'summary.json').read_text())
    lines = ['# Recomputed FrameFlow pilot', '',
        'Scores were recomputed from the source backbones, candidate FASTA files',
        'and saved ESMFold predictions. This analysis makes no API requests.', '',
        'Evaluated backbones: {}/{}. Scored predictions: {}/{}.'.format(
            summary['complete_backbones'], summary['attempted_backbones'],
            summary['successful_predictions'], summary['attempted_sequences']), '',
        'scRMSD is the lowest full-residue CA RMSD among the two designed sequences,',
        'after a proper rigid alignment. Lower is better; the success threshold is 2 A.', '',
        '| Network calls | Method | scRMSD < 2 A | Median scRMSD (A) | Mean scRMSD (A) | Median scTM |',
        '|---:|---|---:|---:|---:|---:|']
    def value(number):
        return 'missing' if number is None else '{:.3f}'.format(number)
    for group in summary['groups']:
        lines.append('| {} | {} | {}/{} | {} | {} | {} |'.format(group['budget'],
            group['method'], group['designable_count'], group['complete_backbones'],
            value(group['median_scRMSD_A']), value(group['mean_scRMSD_A']), value(group['median_scTM'])))
    lines += ['', '## Paired comparisons', '']
    for budget in sorted({p['budget'] for p in summary['paired']}):
        pairs = [p for p in summary['paired'] if p['budget'] == budget and p['complete']]
        count = sum(p['midpoint_minus_euler_scRMSD_A'] < 0 for p in pairs)
        lines.append('- {} calls: midpoint has lower scRMSD in {}/{} complete pairs.'.format(budget, count, len(pairs)))
    lines += ['', '![Paired quality](paired_quality.png)', '',
        'This small pilot does not establish a general solver ranking.',
        'Mean and median scores can favour different methods because of individual poor backbones.', '']
    if (root/'numerical_reference.json').exists():
        numerical = json.loads((root/'numerical_reference.json').read_text())
        lines += ['## Separate numerical study', '',
            'Self-conditioning is disabled and integration stops at t=0.99.',
            'These deviations concern a different experiment from the quality scores.', '']
        for item in numerical['seeds']:
            lines.append('- Seed {}: {} midpoint intervals; reference accepted: **{}**.'.format(
                item['seed'], item['reference_intervals'], item['reference_confirmed_within_tolerances']))
        lines += ['', 'Unconfirmed reference deviations are provisional, not established errors.', '',
            '![Numerical deviations](numerical_deviation.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', root/'REPORT.md')


if __name__ == '__main__':
    main()
