import argparse
import unittest
from unittest.mock import Mock, patch

from common import host
from common.config import resolve_cpu


class Host(unittest.TestCase):
    def test_linux_keeps_strict_mask_and_rejects_drift(self):
        with patch.object(host, 'is_macos', return_value=False), \
             patch.object(host.os, 'sched_getaffinity', return_value={3, 7}, create=True), \
             patch.object(host.os, 'sched_setaffinity', create=True) as pin:
            self.assertEqual(resolve_cpu(argparse.ArgumentParser()), 3)
            host.bind_cpu(7)
            pin.assert_called_once_with(0, {7})
            with self.assertRaisesRegex(RuntimeError, 'affinity changed'):
                host.check_affinity(7)

    def test_macos_qos_does_not_report_a_cpu_mask(self):
        lib = Mock()
        lib.pthread_set_qos_class_self_np.return_value = 0
        lib.qos_class_self.return_value = host.QOS_USER_INTERACTIVE
        with patch.object(host, 'is_macos', return_value=True), \
             patch.object(host, '_darwin', return_value=lib), \
             patch.object(host.os, 'cpu_count', return_value=12):
            self.assertEqual(host.available_cpus(), set(range(12)))
            host.bind_cpu(4)
            lib.pthread_set_qos_class_self_np.assert_called_once_with(host.QOS_USER_INTERACTIVE, 0)
            self.assertIsNone(host.check_affinity(4))
            state = host.scheduling_metadata(4)
            self.assertFalse(state['strict'])
            self.assertIsNone(state['requested_cpu'])
            self.assertEqual(state['runner_slot'], 4)
            lib.qos_class_self.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'QoS changed'):
                host.check_affinity(4)
            lib.pthread_set_qos_class_self_np.return_value = 1
            with self.assertRaises(OSError):
                host.bind_cpu(4)

    def test_arm_native_detection_and_clang_target(self):
        with patch.object(host, 'is_macos', return_value=True), \
             patch.object(host, 'is_arm', return_value=True), \
             patch.object(host, 'cpu_model', return_value='Apple M2 Pro'):
            self.assertEqual(host.resolve_mcpu('native'), 'apple-m2')
            self.assertEqual(host.target_flags('apple-m2'), ['-mcpu=apple-m2'])
            self.assertEqual(host.target_flags('generic'), [])
            self.assertEqual(host.resolve_mcpu('generic'), 'generic')
        with patch.object(host, 'is_arm', return_value=False):
            self.assertEqual(host.target_flags('alderlake'), ['-march=alderlake'])

    def test_iree_darwin_abi_defaults_and_explicit_overrides(self):
        with patch.object(host, 'is_macos', return_value=True), \
             patch.object(host, 'is_arm', return_value=True):
            flags=host.iree_host_flags(['--iree-llvmcpu-enable-ukernels=all'])
            self.assertIn('--iree-llvmcpu-target-triple=arm64-apple-macosx14.0.0',flags)
            self.assertIn('--iree-llvmcpu-link-embedded=false',flags)
            explicit=['--iree-llvmcpu-target-triple=aarch64-none-elf',
                      '--iree-llvmcpu-link-embedded=true','--iree-llvmcpu-system-linker-path=custom']
            self.assertEqual(host.iree_host_flags(explicit),[])
        with patch.object(host, 'is_macos', return_value=False):
            self.assertEqual(host.iree_host_flags([]),[])


if __name__ == '__main__':
    unittest.main()
