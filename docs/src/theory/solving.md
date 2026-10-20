# [6 Solving the Scattering Problem](@id sec-6)

We've noted that the storing the Green function tensor is infeasible, and thus
we resort to using the BTTB structure of the tensor to apply it in
``\order{N\log N}`` time using ``\order{N}`` memory. This is great, but it's not
clear how we can use this to solve [eq. (3.3)](discretization.md#eq-3.3), the discrete
scattering problem ``\vb*{\mathsf{p}}_{\text{d}} = \ab(\bm{\mathsf{I}} -
\bm{\mathsf{X}\mathsf{G}_{\circ}})\vb*{\mathsf{p}}_{\text{s}}``.

## [6.1 Iterative Solvers](@id sec-6.1)

In the spirit of [section 1.6](@ref sec-1.6), we can first approach solving
the scattering problem with only matvecs (a "matvec" being the application of
a matrix to a vector, i.e., the act of computing ``\vb*{\mathsf{f}} =
\bm{\mathsf{G}}_{\circ}\vb*{\mathsf{p}}``) through the so-called Born series.
The Born series is the Taylor (Neumann for matrices) series expansion of the operator:

```@raw html
<a id="eq-6.1"></a>
```
```math
\begin{aligned}
\ab(\bm{\mathsf{I}} - \bm{\mathsf{X}\mathsf{G}_{\circ}})^{-1} &=
\sum_{n=0}^{\infty} \ab(\bm{\mathsf{X}\mathsf{G}_{\circ}})^n = \bm{\mathsf{I}} +
\bm{\mathsf{X}\mathsf{G}_{\circ}} +
\bm{\mathsf{X}\mathsf{G}_{\circ}}\bm{\mathsf{X}\mathsf{G}_{\circ}} +
\dots
\end{aligned}
\tag{6.1}
```

[Eq. (6.1)](#eq-6.1) is technically only convergent if the spectral radius of
``\bm{\mathsf{X}\mathsf{G}_{\circ}}`` is less than 1, that is if the largest
eigenvalue of ``\bm{\mathsf{X}\mathsf{G}_{\circ}}`` is less than 1 (in absolute
value). For most real problems, this is not the case, but if we assume a
material with very weak scattering (i.e., ``\abs{\chi} \ll 1``) so that
``\bm{\mathsf{X}}`` is small enough that ``\bm{\mathsf{X}}\bm{G}_{\circ}`` is
itself small, then the Born series converges and we can use it to solve the
scattering problem:

```math
\begin{aligned}
\tilde{\vb*{\mathsf{p}}}_{\text{s}} &\approx \vb*{\mathsf{p}}_\text{d} + \bm{\mathsf{X}\mathsf{G}_{\circ}}\vb*{\mathsf{p}}_\text{d} + \bm{\mathsf{X}\mathsf{G}_{\circ}}\bm{\mathsf{X}\mathsf{G}_{\circ}}\vb*{\mathsf{p}}_\text{d} + \dots
\end{aligned}
```

where we would keep adding terms until we believe the series has converged. How
do we know when the series has converged though? One way is to check the norm of
the residual. Define the residual vector ``\vb*{\mathsf{r}} =
\vb*{\mathsf{p}}_\text{d} - \ab(\bm{\mathsf{I}}
-\bm{\mathsf{X}\mathsf{G}_{\circ}})\tilde{\vb*{\mathsf{p}}}_\text{s}``. If
``\norm{\vb*{\mathsf{r}}} < \varepsilon`` for some threshold ``\varepsilon``,
then we can say the series has converged and our approximate solution
``\tilde{\vb*{\mathsf{p}}}_\text{s}`` is good enough. If not, we keep adding
terms to the series until it does.

This is an interesting approach, but it will not work in general. For large
``\chi``, or for resonant structures, the Born series will simply not converge.
Even when it does, it may take many iterations to converge, which is not ideal.
One way to interpret the Born series is as a particular (fixed) polynomial over
the of the Krylov space of ``\bm{\mathsf{I}} -
\bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ}``. The Krylov space of a matrix
``\bm{\mathsf{A}}`` is defined as:

```math
\begin{aligned}
\mathcal{K}_k(\bm{\mathsf{A}}, \vb*{\mathsf{b}}) &= \text{span}\left\{\vb*{\mathsf{b}}, \bm{\mathsf{A}}\vb*{\mathsf{b}}, \bm{\mathsf{A}}^2\vb*{\mathsf{b}}, \dots, \bm{\mathsf{A}}^{k-1}\vb*{\mathsf{b}}\right\}
\end{aligned}
```

That is, it is the span of the space spanned by the first ``k`` powers of
``\bm{\mathsf{A}}``. The Born series was the particular polynomial
``p_k\ab(\bm{\mathsf{I}} - \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ}) =
\sum_{n=0}^{k-1} \ab(\bm{\mathsf{X}\mathsf{G}_{\circ}})^n``, setting a
coefficient of 1 for each power of ``\bm{\mathsf{X}\mathsf{G}_{\circ}}`` (i.e.,
each basis vector of the Krylov space). Since our goal is to find a vector
``\vb*{\mathsf{p}}_\text{s}`` such that it's residual ``\vb*{\mathsf{r}} =
\vb*{\mathsf{p}}_\text{d} - \ab(\bm{\mathsf{I}} -
\bm{\mathsf{X}\mathsf{G}_{\circ}})\tilde{\vb*{\mathsf{p}}}_\text{s}`` is small, we can
instead try to find a polynomial ``p_k\ab(\bm{\mathsf{I}} -
\bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ})`` that minimizes the residual norm
``\norm{\vb*{\mathsf{r}}}``. This is the idea behind Krylov subspace methods,
which are a class of iterative methods for solving linear systems that build up a
Krylov space and find the best polynomial over that space to minimize the
residual. The GMRES (Generalized Minimal Residual) method is one such method
that minimizes the inner-product norm of the residual.

```@raw html
<a id="conv-6.1"></a>
```
!!! convention "Gila convention 6.1: Iterative Solvers in Gila"
    In Gila, we expose custom implementations of three iterative solvers: [GMRES](@ref GMRESSolver),
    [GCRO-DR](@ref GCRODRSolver), and [BiCGSTAB](@ref BiCGStabSolver). GCRO-DR is the default solver used in Gila.
    It is a deflated version of GMRES that is more efficient for problems where we change ``\bm{\mathsf{X}}`` slightly (this
    is the solver one might use in inverse design problems). GMRES is the most robust of the three. BiCGSTAB is included,
    but generally does not perform as well as GMRES for typical nanophotonic
    problems. Gila also exposes [`MixPrcRfn`](@ref), a mixed-precision iterative refinement that runs the inner solve in single precision and computes the residual in double precision.

    Technically, Gila's operators are agnostic to the solver used, and any iterative
    solver that only requires matvecs can be used (such as those in
    `IterativeSolvers.jl`). However, the solvers included in Gila are nice
    for the exploration of new preconditioners. Once a robust preconditioner is found,
    these solvers will be removed from Gila.

## [6.2 Preconditioning](@id sec-6.2)

To accelerate the convergence of the iterative solvers, we can use
preconditioners. If we are trying to solve the problem
``\bm{\mathsf{A}}\vb*{\mathsf{x}} = \bm{\mathsf{b}}``, then a left
preconditioner is a matrix ``\bm{\mathsf{M}}_l`` such that we instead solve
``\bm{\mathsf{M}}_l^{-1}\bm{\mathsf{A}}\vb*{\mathsf{x}} =
\bm{\mathsf{M}}_l^{-1}\bm{\mathsf{b}}``. We can also define the right
preconditioner ``\bm{\mathsf{M}}_r`` such that we instead solve
``\bm{\mathsf{A}}\bm{\mathsf{M}}_r^{-1}\vb*{\mathsf{y}} = \bm{\mathsf{b}}``. The
idea of a preconditioner is that we hope to find an operator
``\bm{\mathsf{M}}_l`` or ``\bm{\mathsf{M}}_r`` whose inverse is not only close
to that of the inverse of ``\bm{\mathsf{A}}``, but is also cheap to apply.

The Born series is an example of a preconditioner. In the limit that
``\bm{\mathsf{X}}`` is a weak scatterer, then the operator ``\bm{\mathsf{B}} =
\bm{\mathsf{I}} + \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ}`` is a good
approximation to ``\ab(\bm{\mathsf{I}} - \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ})^{-1}``:

```math
\begin{aligned}
    \bm{\mathsf{B}}\ab(\bm{\mathsf{I}} - \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ}) &= \ab(\bm{\mathsf{I}} + \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ})\ab(\bm{\mathsf{I}} - \bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ}) \\
    &= \bm{\mathsf{I}} - \ab(\bm{\mathsf{X}}\bm{\mathsf{G}}_{\circ})^2
\end{aligned}
```

which is second order in ``\bm{\mathsf{X}}`` (and if ``\bm{\mathsf{X}}`` is
small, then this term is negligible). Unfortunately, the Born series does not
work as an effective preconditioner for most problems, as it is only effective
for weak scatterers, where GMRES already converges quickly.

In Gila, we have yet to find a general preconditioner that works well, but this
is a current active research direction for our group.
