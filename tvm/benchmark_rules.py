"""Benchmark one seeded generation of CPU schedule rules, without search or mutation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load, environment, resolve_cpu
from common.cases import select_cases
from common.operators import inputs, tensor_shapes
from common.host import bind_cpu, shared_suffix, scheduling_metadata, is_macos, linker_flags, native_compiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--cases', help='Comma-separated GEMM names; default: all GEMMs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load(ROOT/'config/common.toml')
    bench = config['benchmark']
    cpu = resolve_cpu(parser, args.cpu)
    os.environ.update(environment(config))
    bind_cpu(cpu)
    import numpy as np
    import tvm
    from tvm.s_tir import meta_schedule as ms
    from common.tvm_ops import module
    from common.runner import check_affinity
    from pytorch.benchmark import PyTorch

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config['benchmark']['cpu'] = cpu
    baseline = PyTorch(config, load(ROOT/'config/pytorch.toml'))
    cases = select_cases(bench, names=args.cases, ops='gemm', root=ROOT)
    target = tvm.target.Target({'kind':'llvm', 'mcpu':bench['mcpu'], 'num-cores':1})
    metadata = dict(tvm=tvm.__version__, target=str(target), cpu=None if is_macos() else cpu, scheduling=scheduling_metadata(cpu), seed=bench['seed'],
                    search_strategy=None, cost_model=None, mutator_probs={}, generations=1,
                    postprocs='from-target for both rule sets', repeats=bench['repeats'],
                    min_repeat_ms=bench['min_repeat_ms'], baseline=baseline.metadata)
    results = []
    def save():
        (out/'results.json').write_text(json.dumps(dict(metadata=metadata, rows=results), indent=2))
    save()
    (out/'run-command.json').write_text(json.dumps([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    for case in cases:
        name = case['name']
        print('CASE', name, case['shape'], flush=True)
        work = out/name
        work.mkdir(exist_ok=True)
        a, b, reference = inputs(case, bench['seed'])
        np.save(work/'A.npy', a); np.save(work/'B.npy', b)
        baseline_dir = work/'baseline'; baseline_dir.mkdir(exist_ok=True)
        baseline_kernel = baseline.prepare(case, a, b, baseline_dir, {})
        baseline_kernel.dump(baseline_dir)
        mod = module(case)
        runtime_a, runtime_b = tvm.runtime.tensor(a), tvm.runtime.tensor(b)
        runtime_c = tvm.runtime.empty(tensor_shapes(case)[2], 'float32')
        candidates = []
        for kind in ['llvm', 'from-target']:
            space = ms.space_generator.PostOrderApply(sch_rules=kind, postprocs='from-target', mutator_probs={})
            context = ms.TuneContext(mod=mod, target=target, space_generator=space,
                                     rand_state=bench['seed'], num_threads=1)
            assert context.search_strategy is None and len(context.space_generator.mutator_probs)==0
            generated = context.space_generator.generate_design_space(mod)
            rules = [type(rule).__name__ for rule in context.space_generator.sch_rules]
            postprocs = [type(proc).__name__ for proc in context.space_generator.postprocs]
            for index, sch in enumerate(generated):
                dest = work/kind/str(index); dest.mkdir(parents=True, exist_ok=True)
                (dest/'raw_tir.py').write_text(sch.mod.script())
                sch.enter_postproc()
                accepted = all(proc.apply(sch) for proc in context.space_generator.postprocs)
                record = dict(name=name, shape=case['shape'], rule_set=kind, candidate=index,
                              generated_candidates=len(generated), rules=rules, postprocs=postprocs,
                              postproc_accepted=accepted, status='postproc_rejected')
                if not accepted:
                    results.append(record); save(); continue
                tir = sch.mod.script()
                (dest/'final_tir.py').write_text(tir)
                (dest/'final_trace.txt').write_text(str(sch.trace))
                record['tir_sha256'] = hashlib.sha256(tir.encode()).hexdigest()
                start = time.perf_counter()
                try:
                    lib = tvm.tirx.build(sch.mod, target=target)
                except Exception as exc:
                    record.update(status='compile_failed', error=str(exc)); results.append(record); save(); continue
                record['compile_s'] = time.perf_counter()-start
                lib.export_library(str(dest/('module'+shared_suffix())), options=linker_flags(), cc=native_compiler('c++'))
                (dest/'kernel.ll').write_text(lib.inspect_source('ll'))
                (dest/'kernel.s').write_text(lib.inspect_source('asm'))
                for _ in range(bench['warmup']):
                    lib(runtime_a, runtime_b, runtime_c); baseline_kernel.run()
                lib(runtime_a, runtime_b, runtime_c)
                np.testing.assert_allclose(runtime_c.numpy(), reference, rtol=bench['rtol'], atol=bench['atol'])
                np.testing.assert_allclose(baseline_kernel.run(), reference, rtol=bench['rtol'], atol=bench['atol'])
                record['max_abs_error'] = float(np.max(np.abs(runtime_c.numpy()-reference)))
                record['observed_affinity'] = check_affinity(cpu)
                record.update(status='prepared', artifact_directory=str(dest))
                candidates.append((lib, record))
                print('PREPARED', name, kind, index, flush=True)
        # Interleave rule sets and alternate baseline order in every timing batch.
        candidates.sort(key=lambda pair:(pair[1]['candidate'], pair[1]['rule_set']))
        for lib, record in candidates:
            backend_times=[]; baseline_times=[]
            evaluator=lib.time_evaluator('main',tvm.cpu(),number=case['reps'],repeat=1,min_repeat_ms=bench['min_repeat_ms'])
            for batch in range(bench['repeats']):
                if batch%2==0:
                    backend_times.append(evaluator(runtime_a,runtime_b,runtime_c).results[0])
                    baseline_times.append(baseline_kernel.measure(bench['min_repeat_ms'],case['reps']))
                else:
                    baseline_times.append(baseline_kernel.measure(bench['min_repeat_ms'],case['reps']))
                    backend_times.append(evaluator(runtime_a,runtime_b,runtime_c).results[0])
                check_affinity(cpu)
            backend_ms=statistics.median(backend_times)*1000
            baseline_ms=statistics.median(baseline_times)*1000
            record.update(status='measured',backend_ms=backend_ms,baseline_ms=baseline_ms,
                          speedup=baseline_ms/backend_ms,backend_seconds=backend_times,baseline_seconds=baseline_times)
            results.append(record); save()
            print('RESULT',name,record['rule_set'],record['candidate'],backend_ms,record['speedup'],flush=True)
        if any(not any(r['status']=='measured' and r['name']==name and r['rule_set']==kind for r in results) for kind in ['llvm','from-target']):
            raise RuntimeError(f'No valid default schedule for {name}')
    table=[]
    lines=['# Default CPU schedule rules without search or mutation','',
           'One design-space generation per case/rule set, fixed seed. No search strategy, cost model, evolutionary iterations or mutators. Both sets use target-selected postprocessing to finalize annotations and isolate schedule-rule differences. All measured candidates pass FP64 correctness checks.','',
           'Rules can sample tiling parameters and produce several candidates. Candidate 0 is listed separately; the best candidate is selected only among this initial generated batch, not through subsequent search.','',
           '| Case | Candidates LLVM / target | First LLVM / target (ms) | Median LLVM / target (ms) | Best LLVM / target (ms) | Same final TIRs |',
           '|---|---:|---:|---:|---:|---|']
    for case in cases:
        row={'name':case['name']}
        groups={kind:sorted([r for r in results if r['name']==case['name'] and r['rule_set']==kind and r['status']=='measured'],key=lambda r:r['candidate']) for kind in ['llvm','from-target']}
        for kind, group in groups.items():
            row.update({kind+'_candidates':len(group),kind+'_first_ms':group[0]['backend_ms'],
                        kind+'_median_ms':statistics.median(r['backend_ms'] for r in group),kind+'_best_ms':min(r['backend_ms'] for r in group)})
        row['identical_final_tirs']=[r['tir_sha256'] for r in groups['llvm']]==[r['tir_sha256'] for r in groups['from-target']]
        table.append(row)
        pairs=[f"{row['llvm_'+field]:.6f} / {row['from-target_'+field]:.6f}" for field in ['first_ms','median_ms','best_ms']]
        lines.append('| '+case['name']+' | '+f"{row['llvm_candidates']} / {row['from-target_candidates']}"+' | '+' | '.join(pairs)+' | '+str(row['identical_final_tirs'])+' |')
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader();writer.writerows(table)
    lines+=['','CPU '+str(cpu)+', target '+bench['mcpu']+'. The separate long tuning experiment ran concurrently on other cores, so cache/memory bandwidth were shared. Each candidate has a paired PyTorch baseline in results.json.','']
    (out/'report.md').write_text('\n'.join(lines))
    manifest=[]
    for path in out.rglob('*'):
        if path.is_file() and path.name!='manifest.json':
            with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            manifest.append(dict(path=str(path.relative_to(out)),sha256=digest,bytes=path.stat().st_size))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',len(cases),'cases',sum(r['status']=='measured' for r in results),'candidates',flush=True)


if __name__=='__main__':main()
