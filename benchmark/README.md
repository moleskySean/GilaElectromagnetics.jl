# Benchmark suite

A repeatable BenchmarkTools suite measuring **creation** (`GlaOprVac` /
`GlaVacOprMem`) and **application** (`opr * vec`) of the vacuum Green operator,
on CPU (and GPU when `CUDA.functional()`).

## Running

```
julia --project=. -t auto benchmark/runBmk.jl         # full reference run (slow)
julia --project=. -t auto benchmark/runBmk.jl --quick # small sizes, for iterating
julia --project=. -t auto benchmark/runBmk.jl --big   # adds application-only (48,64) self sizes
```

The scripts manage their own environment (`benchmark/Project.toml`). The first
run dev-links the parent package and instantiates. Results are saved in
`benchmark/results/` as three files named `bmk_<commit>[-dirty]_<timestamp>`:

- `.json`: BenchmarkTools results (the input for `cmpBmk.jl`)
- `.meta.json`: commit + dirty flag, Julia/CPU/GPU, thread counts (Julia, FFTW, BLAS)
- `.jld2`: results and metadata together

## Comparing runs

```
julia --project=. benchmark/cmpBmk.jl <old>.json <new>.json [--tolerance=0.05]
```

Prints `BenchmarkTools.judge` ratios of the medians per benchmark and exits
non-zero on any time regression. A baseline run should be recorded before any
code changes.

## Plotting

```
julia --project=. benchmark/pltBmk.jl [result.json]
```

Creation and application time versus cubic self-operator size, CPU and GPU
overlaid, saved to `benchmark/bmk.png`. Defaults to the newest result file.

## What is measured

Suite keys: `<create|apply>/<cpu|gpu>/self/<NxNxN>` and
`<create|apply>/<cpu|gpu>/ext/<touching|close|mid|far>/<NxNxN>`.

- **Self operators**: cubic `(n,n,n)`, `n ∈ {4, 8, 16, 24, 32}`, cell scale
  `1//32`, plus the non-cubic `(32,16,8)` to catch FFT anisotropy.
- **External operators**: same-size cubic pairs (`n ∈ {8, 16, 24}`, aligned
  grids), separated along x by a face-to-face gap: `touching` (0 cells, the
  `genCntVol` + `egoFunExtCnt!` contact path), `close` (1 cell, near-singular
  cubature), `mid` (`n` cells), `far` (`20n` cells, FFT-dominated).
- Creation uses `samples=5, evals=1, gcsample=true`. Trial memory/allocations
  are recorded. GPU application benchmarks `CUDA.@sync(opr * vec)` with the
  input already a `CuArray` (i.e., we don't measure the implicit host→ device
  copy path).

The `touching` creation benchmark is built at the memory layer
(`GlaVacOprMem(kerOpt, trgVol, srcVol)`) because `GlaOprVac(trgVol, srcVol)`
reroutes volumes in face contact through the union/self path and would never
exercise the contact fill.
