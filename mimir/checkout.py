"""Reuse one remote checkout and incrementally build it under the shared build root."""
import fcntl
import json
import os
from pathlib import Path
import subprocess

from common.config import ROOT
from common.host import bind_build_cpus, is_macos, linker_flags


def remote_url(remote):
    if '://' not in remote and not remote.startswith(('git@','/','.')) and remote.count('/')==1:
        return 'https://github.com/'+remote.removesuffix('.git')+'.git'
    return remote


def git(source, *args):
    return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()


def checkout(source, remote, ref):
    remote=remote_url(remote)
    if source.exists() and not (source/'.git').exists():
        raise ValueError(f'{source} is not a managed Git checkout; choose another build directory')
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone',remote,str(source)],check=True)
    else:
        # Do not discard edits in a checkout the user may have inspected or modified.
        if git(source,'status','--porcelain','--untracked-files=normal'):
            raise ValueError(f'MimIR checkout has local changes: {source}; commit or remove them before switching')
        if git(source,'remote','get-url','origin')!=remote:
            subprocess.run(['git','-C',str(source),'remote','set-url','origin',remote],check=True)
    subprocess.run(['git','-C',str(source),'fetch','origin',ref],check=True)
    revision=git(source,'rev-parse','FETCH_HEAD')
    subprocess.run(['git','-C',str(source),'switch','--detach',revision],check=True)
    subprocess.run(['git','-C',str(source),'submodule','sync','--recursive'],check=True)
    subprocess.run(['git','-C',str(source),'submodule','update','--init','--recursive'],check=True)
    return revision


def prepare(config, settings):
    build_root=Path(os.environ.get('GEMM_BUILD_DIR',config.get('build',{}).get('directory','build')))
    if not build_root.is_absolute():build_root=ROOT/build_root
    workspace=build_root.resolve()/settings.get('name','mimir');workspace.mkdir(parents=True,exist_ok=True)
    source,build=workspace/'source',workspace/'compiler'
    with (workspace/'checkout.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        revision=checkout(source,settings['remote'],settings['ref'])
        host_flags=[]
        llvm=Path('/opt/homebrew/opt/llvm/bin')
        if is_macos() and (llvm/'clang++').is_file():
            host_flags=[f'-DCMAKE_CXX_COMPILER={llvm/"clang++"}',f'-DCMAKE_C_COMPILER={llvm/"clang"}',f'-DMIM_CLANG={llvm/"clang"}',
                        *[f'-DCMAKE_{kind}_LINKER_FLAGS={" ".join(linker_flags())}' for kind in ['EXE','SHARED','MODULE']]]
        command=[settings['cmake'],'-S',str(source),'-B',str(build),'-G','Ninja',
                 '-DCMAKE_BUILD_TYPE=Release','-DMIM_BUILD_PYTHON=OFF','-DBUILD_TESTING=OFF',*host_flags,*settings['cmake_flags']]
        commands=[command,[settings['cmake'],'--build',str(build),'--target','mim','mim_all_plugins','-j',str(settings['jobs'])]]
        (workspace/'build_commands.json').write_text(json.dumps(commands,indent=2))
        build_cpus={int(cpu) for cpu in os.environ.get('GEMM_MIMIR_BUILD_CPUS','').split(',') if cpu}
        child_setup=(lambda:bind_build_cpus(build_cpus)) if build_cpus and not is_macos() else None
        for name,command in zip(['configure.log','build.log'],commands):
            print(f'MimIR {name}: {workspace/name}',flush=True)
            with (workspace/name).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,preexec_fn=child_setup)
        state=dict(remote=remote_url(settings['remote']),ref=settings['ref'],revision=revision,source=str(source),build=str(build))
        (workspace/'checkout-state.json').write_text(json.dumps(state,indent=2))
    return source,build,state
