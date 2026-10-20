# Benchmarks

`benchmarks.jl` times cold creation of a `GlaOprVac` (one case, which pays for the per-cell-shape table), warm creation, and application, on the CPU and on Metal.

```
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'
benchpkg --rev=main,dirty --add=Metal --exeflags=--threads=auto
benchpkgtable
```

This compares `main` with the working tree. Set `GILA_BENCH_QUICK=1` to skip the 64³ apply.

CI runs the same suite on pull requests to `main` and posts the table in the job summary.

Keys:

```
create/cold/cpu/self/8
create/warm/<dev>/<case>
apply/<dev>/<f32|f64>/<case>
```

`<dev>` is `cpu` or `metal`, and Metal runs f32 only. Timings on shared CI runners are noisy, so trust the memory and allocation ratios.
