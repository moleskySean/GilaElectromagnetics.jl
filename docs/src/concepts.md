# Concepts

GilaElectromagnetics implements the three dimensional electromagnetic Green
function using a volume integral formulation to solve Maxwell's equations
numerically efficiently and accurately. The concepts required for its usage are
presented below.

## Foundations

As goes with most of classical electromagnetics, the starting point is Maxwell's
equations, here presented in their differential form:

```math
\frac{\partial B}{\partial t} + \nabla \times E = 0
```
```math
\frac{\partial D}{\partial t} - \nabla \times H = -J_f
```
```math
\nabla \cdot D = \rho_f
```
```math
\nabla \cdot B = 0
```

The electric and magnetic fields are denoted by ``E`` and ``B`` respectively.
Associated to them are the electric displacement field ``D`` and auxiliary
magnetic field ``B``, connected by their respective constitutive relations:

```math
D = \epsilon E
```
```math
B = \mu H = \mu_0\left(H + M\right)
```

Also present are the free charge density ``\rho_f``, the free current density
``J_f``, the electric permittivity ``\epsilon``, the magnetic permeability
``\mu``, and the magnetization field ``M``.

!!! note "Change to frequency domain"
    GilaElectromagnetics works in the *frequency domain*. As expected, an
    (inverse) Fourier transform is used to rewrite equations in this domain:

    ```math
    f(\omega) = \frac{1}{2\pi}\int_{-\infty}^{\infty} f(t)e^{i\omega t}\,dt
    ```

Of interest for Gila are the first two Maxwell equations presented above,
rewritten in Fourier space and with the constitutive relations:

```math
J_f = i\omega\epsilon E + \nabla \times H
```
```math
-M = -\frac{1}{i\omega\mu_0}\nabla \times E + H
```

This can be compactly written in matrix form:

```math
\begin{pmatrix}
J_f \\
-M
\end{pmatrix}
=
\begin{pmatrix}
i\omega\epsilon\textbf{I}_{3\times 3} & \nabla \times \\
-\frac{1}{i\omega\mu_0} \nabla \times & \textbf{I}_{3 \times 3}
\end{pmatrix}
\begin{pmatrix}
E \\
H
\end{pmatrix}
```

## [Notation and change of coordinates](@id maxwell)

The equation above must be tweaked to have a Hermitian matrix in Fourier space so that it has physical meaning.

!!! note "Hermitian matrix"
    A square matrix ``A`` is said to be *Hermitian* if it is equal to its own
    adjoint (conjugate transpose):
    ```math
    A = A^\dagger = \left(A^\top\right)^* = \left(A^*\right)^\top
    ```
    The eigenvalues of a Hermitian matrix are always real and may represent physical observables.

The following change of coordinates is appropriate for this purpose :

```math
\textbf{j} = J
```
```math
\textbf{m} = i\mu_0\omega M
```
```math
Z = \sqrt{\frac{\mu_0}{\epsilon}}
```
```math
k_0 = \omega\sqrt{\mu_0\epsilon}
```

It leads to the following equation :

```math
\frac{i}{k_0}
\begin{pmatrix}
\textbf{j} \\
-\textbf{m}
\end{pmatrix}
=-
\begin{pmatrix}
Z^{-1} & -\frac{i}{k_0} \nabla \times \\
\frac{i}{k_0} \nabla \times & Z
\end{pmatrix}
\begin{pmatrix}
E \\
H
\end{pmatrix}
```

Finally, the notation is simplified by denoting the vector of ``\textbf{j}`` and
``\textbf{m}`` by ``\textbf{p}``, the vector of fields ``E`` and ``H`` by
``\textbf{f}`` and the matrix, also referred to as the *vacuum Maxwell
operator*, by ``\textbf{M}_0``:

```math
\frac{i}{k_0}\textbf{p} = \textbf{M}_0\textbf{f}
```

This elegantly presents the Maxwell equations in a way that can be approached by
Gila, knowing that the other two Maxwell equations that were not used in the
development are satisfied. See [scattering](./usage.md#scattering).

The Green's function ``\textbf{G}_0`` of the vacuum Maxwell operator is defined
as the inverse of ``\textbf{M}_0``:

```math
\textbf{G}_0\textbf{M}_0 = \textbf{I}_{6\times 6}
```

!!! danger "What GilaElectromagnetics does"
    The purpose of GilaElectromagnetics is to compute the action of this Green's
    function on a vector in vacuum.

Solving for the Green's function in matter can be done indirectly with Gila. See
the next section, [usage](usage.md), for more information.
