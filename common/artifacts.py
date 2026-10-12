"""Export measured artifacts and provenance outside the timing region."""
import hashlib
import json
import shutil


def export_artifacts(kernel, destination, work, metadata, row, baseline_kernel=None):
    destination.mkdir(parents=True,exist_ok=True)
    previous = destination/'manifest.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get('backend') != metadata['backend'] or old.get('shape') != row['shape'] or old.get('op','gemm') != row['op']:
            raise ValueError('Artifact directory belongs to another backend or shape; choose another --dump-dir')
        for key in ['baseline_backend','baseline_config','baseline_metadata','backend_config','device','mcpu','dtype','seed','tvm','git','target','iree_compiler','iree_runtime','pytorch','torch_build','mimir','mimir_git','mimir_build','mimir_clang','mimir_target_flags']:
            if old.get('metadata', {}).get(key) != metadata.get(key):
                raise ValueError(f'Artifact directory contains incompatible {key}; choose another --dump-dir')
    info = kernel.dump(destination)
    if baseline_kernel is not None:
        baseline_destination=destination/'baseline'
        baseline_destination.mkdir(exist_ok=True)
        info['baseline_artifacts']=baseline_kernel.dump(baseline_destination)
    for name in ['A.npy','B.npy']:
        shutil.copy2(work/name,destination/name)
    (destination/'benchmark_result.json').write_text(json.dumps(row,indent=2))
    files = []
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name!='manifest.json':
            with path.open('rb') as source:
                digest = hashlib.file_digest(source,'sha256').hexdigest()
            files.append(dict(path=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=digest))
    manifest = dict(backend=metadata['backend'],device=metadata['device'],
                    op=row['op'],shape=row['shape'],output_shape=row['output_shape'],metadata=metadata,
                    timing_scope=row['timing_scope'],files=files,**info)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
