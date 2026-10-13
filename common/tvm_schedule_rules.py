"""One seeded LLVM-rule generation, candidate zero, no search or mutation."""

from tvm.s_tir import meta_schedule as ms


def schedule(mod, target, seed):
    space = ms.space_generator.PostOrderApply(
        sch_rules=ms.ScheduleRule.create("llvm"),
        postprocs="from-target",
        mutator_probs={},
    )
    context = ms.TuneContext(
        mod=mod, target=target, space_generator=space, rand_state=seed, num_threads=1
    )
    assert context.search_strategy is None
    generator = context.space_generator
    assert generator is not None
    mutators = generator.mutator_probs
    postprocs = generator.postprocs
    rules = generator.sch_rules
    assert mutators is not None and len(mutators) == 0
    assert postprocs is not None and rules is not None
    candidates = generator.generate_design_space(mod)
    if not candidates:
        raise RuntimeError("LLVM rules generated no default schedule")
    selected = candidates[0]
    # Finalize vectorization/unrolling annotations; no search or candidate mutation.
    selected.enter_postproc()
    for postproc in postprocs:
        if not postproc.apply(selected):
            raise RuntimeError(
                f"First LLVM-rule candidate rejected by {type(postproc).__name__}"
            )
    details = dict(
        schedule="llvm",
        schedule_rule="MetaSchedule.DefaultLLVM",
        selected_candidate=0,
        generated_candidates=len(candidates),
        candidate_selection="first; no performance-based selection",
        search_strategy=None,
        mutators=0,
        rules=[type(rule).__name__ for rule in rules],
        postprocs=[type(proc).__name__ for proc in postprocs],
    )
    return selected.mod, str(selected.trace), details
