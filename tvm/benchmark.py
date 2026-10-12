"""TVM adapter; shared workload and BLAS comparison live in common/."""
import os
from pathlib import Path
from importlib.metadata import version
import shutil
import time

import tvm
from common.tvm_ops import module as operator_module
from common.operators import tensor_shapes
from common.tvm_dlight import schedule as dlight_schedule
from common.tvm_schedule_rules import schedule as llvm_schedule
from tvm.s_tir import meta_schedule as ms

from common.runner import run


def builder_affinity():
    cpus = os.environ.get('GEMM_TVM_BUILDER_CPUS')
    if cpus:
        os.sched_setaffinity(0,{int(cpu) for cpu in cpus.split(',')})


class TVM:
    identity_keys = ['tvm','tvm_distribution','target']

    def __init__(self, config, settings):
        self.bench,self.tuning = config['benchmark'],settings['tvm']['auto_tune']
        self.auto_tune = self.tuning.get('enabled',False)
        self.schedule_rules = self.tuning.get('schedule_rules','from-target')
        if self.schedule_rules not in ('from-target','llvm'):
            raise ValueError('tvm.auto_tune.schedule_rules must be from-target or llvm')
        self.schedule_mode = settings['tvm'].get('schedule','llvm')
        if self.schedule_mode not in ('llvm','dlight','naive'):
            raise ValueError('tvm.schedule must be llvm, dlight or naive')
        if self.auto_tune and self.tuning.get('builder_cpus'):
            os.environ['GEMM_TVM_BUILDER_CPUS'] = ','.join(map(str,self.tuning['builder_cpus']))
        assert tvm.runtime.num_threads()==1
        self.target = tvm.target.Target({'kind':'llvm','mcpu':self.bench['mcpu'],'num-cores':1})
        mode = 'auto_tuned' if self.auto_tune else ('llvm_default' if self.schedule_mode == 'llvm' else self.schedule_mode)
        search = 'MetaSchedule EvolutionarySearch + RandomModel' if self.auto_tune else {
            'llvm': 'none: one seeded LLVM-rule generation, candidate 0, no mutations',
            'dlight': 'CPU DLight rules + LLVM',
            'naive': 'none: original operator TIR + default LLVM lowering',
        }[self.schedule_mode]
        effective_rules = self.schedule_rules if self.auto_tune else ('llvm' if self.schedule_mode == 'llvm' else self.schedule_mode)
        self.metadata = dict(schedule_rules=effective_rules,auto_tune=self.auto_tune,variant=mode,tvm=tvm.__version__,tvm_threads=tvm.runtime.num_threads(),
                             target=str(self.target),search=search,
                             trials=self.tuning['trials'] if self.auto_tune else 0,
                             tvm_distribution=version('apache-tvm'),tvm_package_path=str(Path(tvm.__file__).resolve()),
                             compiler_build=tvm.support.libinfo())

    def prepare(self, case, a, b, work, previous):
        mod = operator_module(case)
        if self.auto_tune:
            start = time.perf_counter()
            records = work/'database_tuning_record.json'
            cached = records.exists() and len(records.read_text().splitlines())>=self.tuning['trials']
            if cached:
                database = ms.database.JSONDatabase(work_dir=str(work))
            else:
                space = ms.space_generator.PostOrderApply(
                    sch_rules=ms.ScheduleRule.create('llvm') if self.schedule_rules == 'llvm' else 'from-target')
                database = ms.tir_integration.tune_tir(space=space,mod=mod,target=self.target,work_dir=str(work),
                        max_trials_global=self.tuning['trials'],num_trials_per_iter=self.tuning['num_trials_per_iter'],
                        builder=ms.builder.LocalBuilder(max_workers=self.tuning['builder_workers'],initializer=builder_affinity),
                        runner=ms.runner.LocalRunner(evaluator_config=ms.runner.EvaluatorConfig(
                            number=self.tuning['number'],repeat=self.tuning['repeat'],min_repeat_ms=self.tuning['min_repeat_ms'])),
                        cost_model='random',num_tuning_cores=1,seed=self.bench['seed'])
            tuning_s = previous.get('tuning_s') if cached else time.perf_counter()-start
            schedule = ms.tir_integration.compile_tir(database,mod,self.target)
            if schedule is None:
                raise RuntimeError(f'No valid TVM schedule for {case}')
            final_mod,trace = schedule.mod,str(schedule.trace)
            details = dict(auto_tune=True,tuning_s=tuning_s,valid_records=len(database))
        elif self.schedule_mode == 'llvm':
            final_mod,trace,details = llvm_schedule(mod,self.target,self.bench['seed'])
            details.update(auto_tune=False,tuning_s=0.0,valid_records=0)
        elif self.schedule_mode == 'dlight':
            final_mod,trace,rule = dlight_schedule(mod,self.target,case)
            details = dict(auto_tune=False,tuning_s=0.0,valid_records=0,schedule='dlight',schedule_rule=rule)
        else:
            final_mod,trace = mod,'# Auto tuning disabled; no schedule transformations applied.\n'
            details = dict(auto_tune=False,tuning_s=0.0,valid_records=0)
        (work/'final_tir.py').write_text(final_mod.script())
        (work/'final_trace.txt').write_text(trace)
        if self.auto_tune:
            shutil.copy2(work/'final_tir.py',work/'best_schedule.py')
            shutil.copy2(work/'final_trace.txt',work/'best_trace.txt')
        start = time.perf_counter()
        lib = tvm.tirx.build(final_mod,target=self.target)
        lib.export_library(str(work/'module.so'))
        details['compile_s'] = time.perf_counter()-start
        if self.auto_tune:
            shutil.copy2(work/'module.so',work/'best.so')
        return Kernel(lib,a,b,tensor_shapes(case)[2],work,details)



class Kernel:
    scope = 'native TVM time_evaluator; preallocated output; no Python calls or input copies in timing'

    def __init__(self, lib, a, b, shape, work, details):
        self.work = work
        self.lib,self.details = lib,details
        self.a,self.b = tvm.runtime.tensor(a),tvm.runtime.tensor(b)
        self.c = tvm.runtime.empty(shape,dtype='float32')

    def dump(self, destination):
        names = ['final_tir.py','final_trace.txt','module.so']
        if self.details['auto_tune']:
            names.extend(['best_schedule.py','best_trace.txt','best.so'])
        for name in names:
            shutil.copy2(self.work/name,destination/name)
        (destination/'kernel.ll').write_text(self.lib.inspect_source('ll'))
        (destination/'kernel.s').write_text(self.lib.inspect_source('asm'))
        return dict(notes='Final TIR, optimized LLVM IR, assembly and exported module are from the measured TVM build. Auto tuning: '+str(self.details['auto_tune']))

    def run(self):
        self.lib(self.a,self.b,self.c)
        return self.c.numpy()

    def measure(self, min_repeat_ms, min_reps=1):
        evaluator = self.lib.time_evaluator('main',tvm.cpu(),number=min_reps,repeat=1,min_repeat_ms=min_repeat_ms)
        return evaluator(self.a,self.b,self.c).results[0]


if __name__=='__main__':
    run('tvm',TVM)
