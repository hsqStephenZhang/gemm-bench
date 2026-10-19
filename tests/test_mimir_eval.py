import json
from pathlib import Path
import tempfile
import tomllib
import unittest

from common.config import ROOT
from mimir_eval import configure, report

SPEC=dict(candidate=dict(remote='me/MimIR',ref='abc',label='cand'),baseline=dict(remote='mimir/mimir',ref='def',label='base'))


def results(path, backend, rows, sha):
    path.mkdir(parents=True)
    (path/'results.json').write_text(json.dumps(dict(metadata=dict(backend=backend,mimir_git=sha,cpu='test',mcpu='x'),rows=rows)))


def row(name, op, gflops, speedup):
    return dict(name=name,op=op,shape=[2,2,2],backend_gflops=gflops,baseline_gflops=gflops/speedup,speedup=speedup)


class MimIREval(unittest.TestCase):
    def test_configure_replaces_only_remote_and_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=configure(ROOT/'config/mimir-par.toml','me/MimIR','abc123',Path(tmp)/'c.toml')
            settings=tomllib.loads(out.read_text())['mimir'];original=tomllib.loads((ROOT/'config/mimir-par.toml').read_text())['mimir']
            self.assertEqual((settings['remote'],settings['ref']),('me/MimIR','abc123'))
            self.assertEqual({k:v for k,v in settings.items() if k not in ('remote','ref')},{k:v for k,v in original.items() if k not in ('remote','ref')})

    def test_report_ratios_geomean_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            results(out/'mimir','mimir',[row('a','gemm',10,1.0),row('b','gemm',40,1.0),row('c','gemv',5,1.0)],'d'*40)
            results(out/'mimir-par','mimir-par',[row('a','gemm',20,2.0),row('b','gemm',20,0.5),dict(name='c',op='gemv',shape=[2,2],error='boom')],'a'*40)
            text,failures=report(SPEC,out,60)
            self.assertEqual(failures,{'baseline':[],'candidate':['c']})
            self.assertIn('| gemm | 2 | **1.00** |',text)  # geomean of 2.0 and 0.5
            self.assertIn('| gemv | c | [2, 2] | 5.0 | 5.0 (×1.00) | FAIL | FAIL |',text)
            self.assertIn('`aaaaaaaaaaaa`',text)
            self.assertTrue((out/'comparison.csv').is_file())


if __name__=='__main__':
    unittest.main()
