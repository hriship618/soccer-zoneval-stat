from __future__ import annotations

import argparse
import json
from pathlib import Path

from zcpv.dfl import crunch_match, learn_zone_values


def main():
    parser=argparse.ArgumentParser(description='Compute ZCPV from the official DFL/IDSSE release.')
    parser.add_argument('--data-root',type=Path,default=Path('data/raw/dfl'))
    parser.add_argument('--match',default='J03WMX')
    parser.add_argument('--output',type=Path,default=Path('public/data/dfl-match.json'))
    parser.add_argument('--all',action='store_true',help='Process every complete match and write a combined dataset')
    args=parser.parse_args()
    event_paths=sorted(args.data_root.glob('*/events.xml'))
    if not event_paths: raise SystemExit('No DFL event files found')
    zone_values=learn_zone_values(event_paths)
    if args.all:
        results=[]
        for match_dir in sorted(args.data_root.iterdir()):
            if all((match_dir/name).exists() for name in ('match.xml','events.xml','positions.xml')):
                print(f"Crunching {match_dir.name}...",flush=True)
                results.append(crunch_match(match_dir,event_paths,zone_values))
        result={'provenance':{'source':'DFL / IDSSE','license':'CC BY 4.0','doi':'10.1038/s41597-025-04505-y'},'matches':results}
    else:
        result=crunch_match(args.data_root/args.match,event_paths,zone_values)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    count=sum(len(m['players']) for m in result['matches']) if args.all else len(result['players'])
    print(f"Wrote {args.output} with {count} player rows")

if __name__=='__main__': main()
