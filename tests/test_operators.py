import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock

import numpy as np
import torch

from common.artifacts import export_artifacts
from common.cases import load_cases,select_cases
from common.config import ROOT,load
from common.operators import flops,inputs,tensor_shapes,work_name
from common.report import summarize


class Operators(unittest.TestCase):
    def test_references_and_shapes(self):
        torch.set_num_threads(1)
        cases=[dict(op='gemm',shape=[5,7,3]),dict(op='gemv',shape=[5,3]),
               dict(op='conv2d',shape=[2,3,9,11,5,3,2,2,1]),dict(op='conv2d',shape=[1,2,5,7,3,1,1,1,0])]
        for case in cases:
            with self.subTest(case=case):
                a,b,ref=inputs(case,42);x,y=torch.from_numpy(a).double(),torch.from_numpy(b).double()
                if case['op']=='conv2d':out=torch.nn.functional.conv2d(x,y,stride=case['shape'][7],padding=case['shape'][8]).numpy()
                else:out=(x@y).numpy()
                np.testing.assert_allclose(ref,out,rtol=1e-12,atol=1e-12)
                self.assertEqual(out.shape,tensor_shapes(case)[2])
                self.assertGreater(flops(case),0)

    def test_case_formats_selection_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cases.txt'
            path.write_text('gemm g 5 3 7 2\ngemv v 5 3 2\nconv c 1 2 9 11 5 3 2 2 1 2\n')
            cases=load_cases(path)
            self.assertEqual([c['op'] for c in cases],['gemm','gemv','conv2d'])
            self.assertEqual(cases[0]['shape'],[5,7,3])
            self.assertEqual(select_cases({},case_file=path,names='c,v',ops='gemv,conv2d',reps=3)[0]['reps'],3)
            self.assertEqual(select_cases({},shapes='5,7,3',ops='gemm')[0]['shape'],[5,7,3])
            for line in ['conv2d bad 1 2 2 2 3 5 5 1 0 1','conv2d bad 1 2 5 5 3 3 3 0 0 1','gemv bad 5 0 1','conv2d bad 1 2 5 5 3 3 3 1 -1 1']:
                path.write_text(line)
                with self.subTest(line=line),self.assertRaises(ValueError):load_cases(path)

    def test_default_suite_and_operator_directories(self):
        cases=select_cases(load(ROOT/'config/common.toml')['benchmark'],root=ROOT)
        self.assertEqual(len(cases),32)
        self.assertEqual(len({work_name(c) for c in cases}),32)
        self.assertEqual(work_name(dict(op='gemm',shape=[5,7,3])),'5_7_3')
        self.assertEqual(work_name(dict(op='gemv',shape=[5,3])),'gemv_5_3')

    def test_mandatory_torch_and_cli_override(self):
        spec=importlib.util.spec_from_file_location('entry',ROOT/'run.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'common.toml';cfg.write_text((ROOT/'config/common.toml').read_text().replace('tvm = true','tvm = false').replace('iree = true','iree = false'))
            for flags,expected in [([],['pytorch','openblas']),(['--backends','none'],['pytorch']),(['--backends','tvm,iree'],['pytorch','tvm','iree'])]:
                with self.subTest(flags=flags),patch.object(sys,'argv',['run.py','--common-config',str(cfg),'--output',tmp,*flags]),patch.object(entry.subprocess,'run') as run,patch.object(entry,'summarize') as summary:
                    entry.main()
                    self.assertEqual([Path(c.args[0][c.args[0].index('--output')+1]).name for c in run.call_args_list],expected)
                    for call in run.call_args_list:self.assertEqual(call.kwargs['env']['GEMM_PYTORCH_CONFIG'],str(ROOT/'config/pytorch.toml'))
                    self.assertEqual(summary.call_args.args[0],[Path(tmp)/b/'results.json' for b in expected])

    def test_artifact_guard_precedes_baseline_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)
            (destination/'manifest.json').write_text(json.dumps(dict(backend='tvm',op='gemv',shape=[5,3],metadata=dict(baseline_backend='openblas'))))
            kernel,baseline=Mock(),Mock()
            with self.assertRaises(ValueError):
                export_artifacts(kernel,destination,destination,dict(backend='tvm',baseline_backend='pytorch'),dict(op='gemv',shape=[5,3]),baseline)
            kernel.dump.assert_not_called();baseline.dump.assert_not_called()

    def test_historical_report_keeps_baseline_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for backend,new in [('tvm',False),('iree',True)]:
                row=dict(name='example',m=3,n=5,k=7,device='cpu',backend_ms=2,backend_gflops=1,speedup=2,timing_scope='test')
                row.update(dict(baseline_backend='pytorch',baseline_ms=4,baseline_gflops=.5) if new else dict(openblas_ms=4,openblas_gflops=.5))
                p=root/(backend+'.json');p.write_text(json.dumps(dict(metadata=dict(backend=backend),rows=[row])));paths.append(p)
            rows=summarize(paths,root/'summary.csv')
            self.assertEqual([r['baseline_backend'] for r in rows],['openblas','pytorch'])
            self.assertEqual([r['speedup'] for r in rows],[2,2])


if __name__=='__main__':unittest.main()
