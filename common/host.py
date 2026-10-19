"""CPU scheduling and native toolchain differences between Linux and macOS.

Darwin QoS is a scheduling request, never a CPU mask or a core assignment.
"""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def is_macos():
    return sys.platform == 'darwin'


def is_arm():
    return platform.machine().lower() in ('arm64', 'aarch64')


def available_cpus():
    if not is_macos():
        return set(os.sched_getaffinity(0))
    # On Darwin these IDs are runner slots, not an affinity mask.
    return set(range(os.cpu_count() or 1))


def sysctl(name):
    return subprocess.check_output(['/usr/sbin/sysctl', '-n', name], text=True).strip()


def cpu_model():
    if is_macos():
        return sysctl('machdep.cpu.brand_string')
    return next((line.split(':', 1)[1].strip()
                 for line in Path('/proc/cpuinfo').read_text().splitlines()
                 if line.startswith('model name')), platform.machine())


def resolve_mcpu(value):
    if value != 'native':
        return value
    if is_macos() and is_arm():
        match = re.search(r'Apple M(\d+)', cpu_model())
        if not match:
            raise ValueError('Cannot identify Apple CPU; specify --mcpu explicitly')
        return 'apple-m' + match[1]
    # Not every consumer accepts "native" (TVM's LLVM silently falls back to generic on x86),
    # so resolve the host CPU name once, as Clang's -march/-mcpu=native would.
    return host_llvm_cpu() or 'native'


@lru_cache(maxsize=None)
def host_llvm_cpu():
    clang = next((c for c in ['clang','/opt/homebrew/opt/llvm/bin/clang'] if shutil.which(c)), None)
    if clang is None:
        return None
    flag = '-mcpu=native' if is_arm() else '-march=native'
    result = subprocess.run([clang,'-###',flag,'-x','c','-c','/dev/null','-o','/dev/null'],capture_output=True,text=True)
    match = re.search(r'"-target-cpu" "([^"]+)"',result.stderr)
    return match[1] if match else None


def target_flags(mcpu):
    if mcpu == 'generic':
        return []
    return [f"-{'mcpu' if is_arm() else 'march'}={mcpu}"]


def shared_suffix():
    return '.dylib' if is_macos() else '.so'


def linker_flags():
    # Homebrew LLVM's bundled linker may not understand the newest Apple SDK.
    if is_macos():
        return ['-fuse-ld=/Library/Developer/CommandLineTools/usr/bin/ld',
                '-isysroot', '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk']
    return []


def native_compiler(compiler):
    # An older selected Xcode toolchain can be incompatible with the CLT SDK.
    if is_macos() and compiler in ('cc', 'c++', 'clang', 'clang++'):
        name = 'clang++' if compiler in ('c++', 'clang++') else 'clang'
        llvm = Path('/opt/homebrew/opt/llvm/bin')/name
        if llvm.is_file():
            return str(llvm)
        tool = Path('/Library/Developer/CommandLineTools/usr/bin')/name
        if tool.is_file():
            return str(tool)
    return compiler


@lru_cache(maxsize=1)
def _darwin():
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    lib.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
    lib.pthread_set_qos_class_self_np.restype = ctypes.c_int
    lib.qos_class_self.restype = ctypes.c_uint
    return lib


QOS_USER_INTERACTIVE = 0x21


def bind_cpu(cpu):
    if not is_macos():
        os.sched_setaffinity(0, {cpu})
        return
    result = _darwin().pthread_set_qos_class_self_np(QOS_USER_INTERACTIVE, 0)
    if result:
        raise OSError(result, 'Cannot set macOS benchmark thread QoS')


def bind_build_cpus(cpus):
    # Darwin leaves compiler workers to the scheduler; do not pretend to pin them.
    if cpus and not is_macos():
        os.sched_setaffinity(0, set(cpus))


def check_affinity(cpu):
    if is_macos():
        actual = _darwin().qos_class_self()
        if actual != QOS_USER_INTERACTIVE:
            raise RuntimeError(f'Runner QoS changed: expected {QOS_USER_INTERACTIVE}, observed {actual}')
        return None
    actual = os.sched_getaffinity(0)
    if actual != {cpu}:
        raise RuntimeError(f'Runner CPU affinity changed: expected {[cpu]}, observed {sorted(actual)}')
    return sorted(actual)


def scheduling_metadata(cpu):
    if is_macos():
        return dict(mode='macos-qos', strict=False, requested_cpu=None,
                    runner_slot=cpu, qos='user-interactive',
                    observed_qos=_darwin().qos_class_self(),
                    note='OS selects cores dynamically; no CPU affinity or core isolation')
    return dict(mode='linux-affinity', strict=True, requested_cpu=cpu,
                observed_affinity=check_affinity(cpu))


def configure_openblas(settings):
    if not is_macos() or settings['library'] != 'auto':
        return
    prefixes = [Path('/opt/homebrew/opt/openblas'), Path('/usr/local/opt/openblas')]
    prefix = next((p for p in prefixes if (p/'lib/libopenblas.dylib').is_file()), None)
    if prefix is None:
        raise ValueError('OpenBLAS missing; install with brew install openblas or configure its library/flags')
    settings['library'] = str(prefix/'lib/libopenblas.dylib')
    settings['compile_flags'] += ['-I', str(prefix/'include')]
    settings['link_flags'] = ['-L', str(prefix/'lib'), f'-Wl,-rpath,{prefix/"lib"}', *settings['link_flags']]


def configure_host(config):
    config['benchmark']['mcpu'] = resolve_mcpu(config['benchmark'].get('mcpu', 'native'))
    if not is_macos() and config['openblas']['library'] == 'auto':
        config['openblas']['library'] = 'libopenblas.so'


def iree_host_flags(flags):
    if not is_macos():
        return []
    defaults = {
        '--iree-llvmcpu-target-triple': f"{'arm64' if is_arm() else 'x86_64'}-apple-macosx14.0.0",
        '--iree-llvmcpu-link-embedded': 'false',
        '--iree-llvmcpu-system-linker-path': '/Library/Developer/CommandLineTools/usr/bin/ld',
    }
    # Native Mach-O uses Darwin's ABI for the optimized ARM microkernels.
    return [key+'='+value for key, value in defaults.items()
            if not any(flag.split('=', 1)[0] == key for flag in flags)]


def disassemble(tool, library, output, symbol=None):
    # Apple's objdump lacks GNU --disassemble=<symbol>; dump the wrapper library.
    command = [tool, '-d']
    if symbol and not is_macos():
        command.append('--disassemble=' + symbol)
    with Path(output).open('w') as out:
        subprocess.run([*command, str(library)], stdout=out, check=True)
