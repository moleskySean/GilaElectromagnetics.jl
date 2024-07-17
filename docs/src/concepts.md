# Concepts

GilaElectromagnetics implements the three dimensionnal electromagnetic Green function with volume integral techniques to solve Maxwell's equations numerically with great speed. The broad concepts required for it's usage are presented in the next subsections.

## Foundations

As goes with most of classical electromagnetics, the starting point is Maxwell's equations, here presented in their differential form :

```math
\frac{\partial B}{\partial t} + \nabla \times E = 0
```
```math
\frac{\partial D}{\partial t} - \nabla \times H = -J
```
```math
\nabla \cdot D = \rho
```
```math
\nabla \cdot B = 0
```

Here, the electric and magnetic fields are denoted by ``E`` and ``H`` respectively. Associated to them are the auxiliary fields ``D`` and ``B``, described by their respective constitutive relations :

```math
D = \epsilon E
```
```math
H = \frac{1}{\mu}B-M
```

Also present are the charge density ``\rho``, the current density ``J``, the electric permittivity ``\epsilon`` and the magnetic permeability ``\mu``.

!!! note "Change to frequency domain"
    GilaElectromagnetics implements a *frequency domain* solver. As expected, the Fourier transform is used to rewrite equations in this domain, also called *Fourier space* :

    ```math
    f(\omega) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(t)e^{-i\omega t}\;dt
    ```

Of interest for Gila are the first two Maxwell equations presented above, rewritten in Fourier space and with constitutive relationships :

```math
J = i\epsilon \omega E + \nabla \times H
```
```math
-M = \frac{i}{\mu \omega}\nabla \times E + H
```

This can be easily represented under matrix form :

```math
\begin{pmatrix}
J \\
-M
\end{pmatrix}
=
\begin{pmatrix}
i\epsilon \omega & \nabla \times \\
\frac{i}{\mu \omega} \nabla \times & 1
\end{pmatrix}
\begin{pmatrix}
E \\
H
\end{pmatrix}
```

## Change of coordinates

The equation above must be tweaked to have a Hermitian matrix in Fourier space so that it has physical meaning.

!!! note "Hermitian matrix"
    A square matrix ``A`` composed of complex elements is said to be *Hermitian* (denoted by subscript ``\dagger``) if it is equal to its own conjugate transform.
    ```math
    A^\dagger = (A^T)^*
    ```
    The eigenvalues of a Hermitian matrix are real and can represent physical observables.
    

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
k_0 = \omega\sqrt{\mu\epsilon}
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

Finally, the notation is simplified by denoting the vector of ``\textbf{j}`` and ``\textbf{m}`` by ``\textbf{p}``, the vector of fields ``E`` and ``H`` by ``\textbf{f}`` and the matrix, also referred to as the *Maxwell operator*, by ``\textbf{M}`` :

```math
\frac{i}{k_0}\textbf{p} = -\textbf{M}\textbf{f}
```

This elegantly presents the Maxwell equations in a way that can be approached by Gila, knowing that the other two Maxwell equations that were not used in the development are satisfied. See scattering

## Solution for vacuum

For the following, *natural units* are used : ``\epsilon_0`` and ``\mu_0`` are set to 1. A Maxwell operator for vacuum ``\textbf{M}_0`` can be defined as :

```math
\textbf{M}_0 = 
\begin{pmatrix}
\textbf{I}_{3\times 3} & -\frac{i}{k_0} \nabla \times \\
\frac{i}{k_0} \nabla \times & \textbf{I}_{3\times 3}
\end{pmatrix}
```

The Green's function ``\textbf{G}_0`` of the vacuum Maxwell operator is defined as such :

```math
\textbf{G}_0\textbf{M}_0 = \textbf{I}_{6\times 6}
```

!!! danger "What GilaElectromagnetics does"
    The main purpose of GilaElectromagnetics is numerically solving for the Green's function of the vacuum Maxwell operator ``\textbf{G}_0`` in a given space.

Matter can be treated using other functionnalities of the package. See ["Usage"](usage.md) for more information.
