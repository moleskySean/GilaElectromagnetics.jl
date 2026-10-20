# [A Fourier Transform Conventions](@id sec-A)

The generic Fourier transform of a function ``f\ab(x)`` is defined as:

```math
\begin{aligned}
    \tilde{f}\ab(\xi) &= \ab[\mathcal{F}\ab(f)]\ab(\xi) \\
    &= \int_{\mathbb{R}} \dd{x} f\ab(x) e^{-2\pi i \xi x}
\end{aligned}
```

and has the inverse transform:

```math
\begin{aligned}
    f\ab(x) &= \ab[\mathcal{F}^{-1}\ab(\tilde{f})]\ab(x) \\
    &= \int_{\mathbb{R}} \dd{\xi} \tilde{f}\ab(\xi) e^{2\pi i \xi x}.
\end{aligned}
```

Unfortunately, there are many different conventions for what ``\xi`` is leading
to many different definitions of the Fourier transform. In these notes, we try
to be self-consistent, and consistent with a broad community of physicists.

Let ``x\mapsto x`` denote a spatial variable and let ``\xi\mapsto k`` be the
associated dual spatial frequency variable. The convention we use in these notes
is that ``k = 2\pi \xi`` such that a spatial domain signal ``f\ab(x)`` has a
Fourier transform ``\tilde{f}\ab(k)`` with:

```math
\begin{aligned}
    \tilde{f}\ab(k) = \int_{\mathbb{R}} \dd{x} f\ab(x) e^{-i k x}, \qquad f\ab(x) = \frac{1}{2\pi}\int_{\mathbb{R}} \dd{k} \tilde{f}\ab(k) e^{i k x}.
\end{aligned}
```

The same convention is taken for the discrete Fourier transform (DFT) and its
inverse (iDFT).

For the temporal transform, we use a different sign. The idea is that we want
plane waves to have the form ``e^{i(\vb*{k}\cdot\vb*{r} - \omega t)}`` in the
time domain, which is the convention used in physics. This means that we let
``x\mapsto t`` be our temporal variable and ``\xi \mapsto \nu`` be the
associated dual temporal frequency variable. We then further define the angular
frequency ``\omega = 2\pi \nu`` such that a time domain signal ``f\ab(t)`` has a
Fourier transform ``\tilde{f}\ab(\omega)`` with:

```math
\begin{aligned}
    \tilde{f}\ab(\omega) = \int_{\mathbb{R}} \dd{t} f\ab(t) e^{i \omega t}, \qquad f\ab(t) = \frac{1}{2\pi}\int_{\mathbb{R}} \dd{\omega} \tilde{f}\ab(\omega) e^{-i \omega t}.
\end{aligned}
```

In the notes, we've dropped the tilde in the frequency domain functions, as we
do not deal with their time domain counterparts in these notes for very long.
