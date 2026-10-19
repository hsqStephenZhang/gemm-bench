"""Compare two MimIR revisions with the gemm-bench harness and write a Markdown report.

The candidate runs as backend `mimir-par`, the baseline as `mimir`, each in its persistent
checkout under build/, one after the other on the same core. The headline per-case figure is
candidate GFLOPS / baseline GFLOPS; each side's speedup over its interleaved PyTorch run is
shown for context (dividing by PyTorch would add PyTorch's own noise on tiny cases).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parent


def configure(template, remote, ref, output):
    """Copy a MimIR backend config with remote/ref replaced (keeps its name and build dir)."""
    text = template.read_text()
    for key,value in [('remote',remote),('ref',ref)]:
        text,count = re.subn(rf'^{key} = .*$',f'{key} = "{value}"',text,count=1,flags=re.M)
        if count!=1:raise ValueError(f'{template} has no {key} setting')
    output.write_text(text);return output


def load(path):
    if not path.is_file():return {},{}
    data = json.loads(path.read_text())
    return data['metadata'],{row['name']:row for row in data['rows']}


def geomean(values):
    return math.exp(sum(map(math.log,values))/len(values)) if values else None


def report(spec, output, elapsed):
    base_meta,base = load(output/'mimir/results.json')
    cand_meta,cand = load(output/'mimir-par/results.json')
    names = list(dict.fromkeys([*cand,*base]))
    rows,ratios,failures = [],{},{'baseline':[],'candidate':[]}
    for name in names:
        b,c = base.get(name,{}),cand.get(name,{})
        op = c.get('op',b.get('op'))
        for side,row in [('baseline',b),('candidate',c)]:
            if 'error' in row or not row:failures[side].append(name)
        ok = 'backend_gflops' in b and 'backend_gflops' in c
        ratio = c['backend_gflops']/b['backend_gflops'] if ok else None
        if ok:ratios.setdefault(op,[]).append(ratio)
        rows.append(dict(op=op,case=name,shape=json.dumps(c.get('shape',b.get('shape'))),
                         pytorch_gflops=c.get('baseline_gflops',b.get('baseline_gflops')),
                         baseline_gflops=b.get('backend_gflops'),candidate_gflops=c.get('backend_gflops'),
                         baseline_vs_pytorch=b.get('speedup'),candidate_vs_pytorch=c.get('speedup'),candidate_vs_baseline=ratio,
                         baseline_error=b.get('error','' if b else 'missing'),candidate_error=c.get('error','' if c else 'missing')))
    with (output/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case']);writer.writeheader();writer.writerows(rows)
    num = lambda v,fmt='.2f':'FAIL' if v is None else format(v,fmt)
    bold = lambda v:'FAIL' if v is None else (f'**{v:.2f}**' if v>=1.0 else f'{v:.2f}')
    revision = lambda meta,side:f"`{spec[side]['remote']}` @ `{(meta.get('mimir_git') or spec[side]['ref'])[:12]}`"
    host = cand_meta or base_meta
    lines = [f"# MimIR eval: {spec['candidate'].get('label','candidate')} vs {spec['baseline'].get('label','baseline')}",'',
             f"- Candidate: {revision(cand_meta,'candidate')}",f"- Baseline: {revision(base_meta,'baseline')}",
             f"- Host: {host.get('cpu','?')}, mcpu `{host.get('mcpu','?')}`, one thread; {elapsed/60:.0f} min",
             '- Ratio = candidate GFLOPS / baseline GFLOPS on the same core; > 1 means the candidate is faster. (×N) = speedup over PyTorch timed interleaved with that side.','',
             '| Op | Cases | Geomean candidate/baseline |','|---|---|---|']
    for op,values in ratios.items():lines.append(f'| {op} | {len(values)} | {bold(geomean(values))} |')
    every = [v for values in ratios.values() for v in values]
    lines += [f'| **all** | {len(every)} | {bold(geomean(every))} |','']
    for side in ['candidate','baseline']:
        if failures[side]:lines.append(f"**{side.capitalize()} failed:** {', '.join(failures[side])}")
    lines += ['','| Op | Case | Shape | PyTorch GFLOPS | Baseline GFLOPS (×PyTorch) | Candidate GFLOPS (×PyTorch) | Candidate/baseline |','|---|---|---|---|---|---|---|']
    side = lambda g,x:'FAIL' if g is None else f'{g:.1f} (×{x:.2f})'
    for row in rows:
        lines.append(f"| {row['op']} | {row['case']} | {row['shape']} | {num(row['pytorch_gflops'],'.1f')} | {side(row['baseline_gflops'],row['baseline_vs_pytorch'])} | {side(row['candidate_gflops'],row['candidate_vs_pytorch'])} | {bold(row['candidate_vs_baseline'])} |")
    text = '\n'.join(lines)+'\n'
    (output/'report.md').write_text(text)
    return text,failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec',type=Path,default=ROOT/'eval/mimir.toml')
    parser.add_argument('--output',type=Path,help='Fresh output root (default: results/mimir-eval-<time>)')
    for side in ['candidate','baseline']:
        parser.add_argument(f'--{side}-remote');parser.add_argument(f'--{side}-ref')
    parser.add_argument('--ops');parser.add_argument('--cases')
    parser.add_argument('--cpu',type=int);parser.add_argument('--common-config',type=Path);parser.add_argument('--build-dir',type=Path)
    parser.add_argument('--report-only',action='store_true',help='Rebuild report.md from an existing --output')
    args = parser.parse_args()
    spec = tomllib.loads(args.spec.read_text())
    for side in ['candidate','baseline']:
        for key in ['remote','ref']:
            if getattr(args,f'{side}_{key}'):spec[side][key]=getattr(args,f'{side}_{key}')
    run = spec.get('run',{})
    output = (args.output or ROOT/'results'/time.strftime('mimir-eval-%Y%m%d-%H%M%S')).resolve()
    start,status = time.time(),0
    if not args.report_only:
        configs = output/'configs';configs.mkdir(parents=True,exist_ok=True)
        (configs/'spec.json').write_text(json.dumps(spec,indent=2))
        base = configure(ROOT/'config/mimir.toml',spec['baseline']['remote'],spec['baseline']['ref'],configs/'baseline.toml')
        cand = configure(ROOT/'config/mimir-par.toml',spec['candidate']['remote'],spec['candidate']['ref'],configs/'candidate.toml')
        command = [sys.executable,str(ROOT/'run.py'),'--backends','mimir,mimir-par','--mimir-config',str(base),'--mimir-par-config',str(cand),
                   '--ops',args.ops or run.get('ops','all'),'--keep-going','--output',str(output)]
        cases = args.cases or run.get('cases','')
        if cases:command += ['--cases',cases]
        for flag,value in [('--cpu',args.cpu),('--common-config',args.common_config),('--build-dir',args.build_dir)]:
            if value is not None:command += [flag,str(value)]
        print('+ '+' '.join(command),flush=True)
        status = subprocess.call(command,cwd=ROOT)
    text,failures = report(spec,output,time.time()-start)
    print(text);print(f'Report: {output/"report.md"}')
    # Candidate failures fail the job; baseline failures are only reported.
    return status or (1 if failures['candidate'] else 0)


if __name__=='__main__':
    sys.exit(main())
