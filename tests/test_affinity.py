import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.config import ROOT,parse_args,resolve_cpu


class Affinity(unittest.TestCase):
    def test_resolve_default_and_explicit_cpu(self):
        parser=argparse.ArgumentParser()
        with patch('common.config.available_cpus',return_value={7,3}):
            self.assertEqual(resolve_cpu(parser),3)
            self.assertEqual(resolve_cpu(parser,7),7)
            with contextlib.redirect_stderr(io.StringIO()) as out,self.assertRaises(SystemExit) as exc:
                resolve_cpu(parser,0)
            self.assertEqual(exc.exception.code,2)
            self.assertIn('CPU 0 is unavailable. Allowed CPUs: [3, 7]',out.getvalue())

    def parse(self, backend='pytorch',flags=(),allowed=(3,7),common=None,settings=None):
        env=dict(GEMM_COMMON_CONFIG=str(common or ROOT/'config/common.toml'),GEMM_BACKEND_CONFIG=str(settings or ROOT/f'config/{backend}.toml'))
        with patch.dict(os.environ,env),patch.object(sys,'argv',['benchmark.py','--cases','mv_pack',*flags]),patch('common.config.available_cpus',return_value=set(allowed)),patch('common.config.bind_cpu') as pin:
            result=parse_args(backend)
            pin.assert_called_once_with(result[0].cpu)
        return result

    def test_shared_parser_selects_allowed_cpu_and_records_mask(self):
        args,config,_,_=self.parse()
        self.assertEqual(args.cpu,3)
        self.assertEqual(config['benchmark']['cpu'],3)
        self.assertEqual(config['benchmark']['allowed_cpus'],[3,7])
        args,_,_,_=self.parse(flags=['--cpu','7'])
        self.assertEqual(args.cpu,7)

    def test_configured_cpu_and_cli_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'common.toml'
            config.write_text((ROOT/'config/common.toml').read_text().replace('[benchmark]','[benchmark]\ncpu = 7'))
            self.assertEqual(self.parse(common=config)[0].cpu,7)
            self.assertEqual(self.parse(common=config,flags=['--cpu','3'])[0].cpu,3)

    def test_unavailable_cpu_fails_before_pin(self):
        env=dict(GEMM_COMMON_CONFIG=str(ROOT/'config/common.toml'),GEMM_BACKEND_CONFIG=str(ROOT/'config/pytorch.toml'))
        with patch.dict(os.environ,env),patch.object(sys,'argv',['benchmark.py','--cpu','0']),patch('common.config.available_cpus',return_value={3,7}),patch('common.config.bind_cpu') as pin,contextlib.redirect_stderr(io.StringIO()) as out,self.assertRaises(SystemExit) as exc:
            parse_args('pytorch')
        self.assertEqual(exc.exception.code,2);pin.assert_not_called()
        self.assertIn('Allowed CPUs: [3, 7]',out.getvalue())

    def test_tvm_builders_use_allowed_cpus_and_single_core_fallback(self):
        _,_,settings,_=self.parse('tvm',flags=['--auto-tune'])
        self.assertEqual(settings['tvm']['auto_tune']['builder_cpus'],[7])
        _,_,settings,_=self.parse('tvm',flags=['--auto-tune'],allowed=(3,))
        self.assertEqual(settings['tvm']['auto_tune']['builder_cpus'],[3])

    def test_tvm_rejects_unavailable_explicit_builder_cpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=Path(tmp)/'tvm.toml'
            settings.write_text((ROOT/'config/tvm.toml').read_text().replace('builder_cpus = []','builder_cpus = [2,4]'))
            env=dict(GEMM_COMMON_CONFIG=str(ROOT/'config/common.toml'),GEMM_BACKEND_CONFIG=str(settings))
            with patch.dict(os.environ,env),patch.object(sys,'argv',['benchmark.py','--auto-tune']),patch('common.config.available_cpus',return_value={3,7}),patch('common.config.bind_cpu') as pin,contextlib.redirect_stderr(io.StringIO()) as out,self.assertRaises(SystemExit):
                parse_args('tvm')
            pin.assert_not_called();self.assertIn('TVM builder CPUs [2, 4] are unavailable',out.getvalue())


if __name__=='__main__':unittest.main()
