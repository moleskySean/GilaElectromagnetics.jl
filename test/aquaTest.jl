using Aqua

@testset "Aqua quality checks" begin
    # `persistent_tasks` is disabled: its probe subprocess fails to precompile
    # when CUDA is in the dependency closure (Aqua reports "done.log was not
    # created, but precompilation exited"). GilaElectromagnetics has no
    # `__init__` and spawns no tasks, so this is a false positive from the
    # check itself rather than a real persistent task. All other checks run.
    Aqua.test_all(GilaElectromagnetics; persistent_tasks=false)
end
