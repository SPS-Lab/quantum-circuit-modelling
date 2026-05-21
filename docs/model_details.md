# Model details

## Effective model

The effective model is

$$
\begin{equation}
H_{\mathrm{eff}}(\phi) = \sum_{j\in\{0,1\}} \frac{\tilde{\omega}_j(\phi)}{2} Z_j
+J(\phi)\bigl(X_1X_0+Y_1Y_0\bigr)
+\frac{\zeta(\phi)}{4} Z_1 Z_0.
\end{equation}
$$

The dressed qubit frequencies are modelled as harmonics

$$
\begin{equation}
\tilde\omega_j(\phi) = 2\pi x_{j,0} + \sum_{k=1}^3 a_{j,k}\cos(2\pi k\phi), \quad j=0,1.
\end{equation}
$$

$J$ and $\zeta$ is modelled with one coefficient set

$$\\{\gamma~,~c_{0}~,~c_{r_0}~,~c_{r_1}~,~c_{\mathrm{prod}}~,~c_{{r_0},\mathrm{sq}}~,~c_{{r_1},\mathrm{sq}}\\}$$

each as

$$
\begin{equation}
O(\phi) = c_{0} + c_{r_0}~r_0(\phi) + c_{r_1}~r_1(\phi) + c_{\mathrm{prod}}~(r_0(\phi)~r_1(\phi)) + c_{r_0,\mathrm{sq}}~r_0(\phi)^2 + c_{r_1,\mathrm{sq}}~r_1(\phi)^2\quad O=J,\zeta
\end{equation}
$$

where

* the residuals $r_j(\phi) = \frac{1}{\sqrt{\Delta_j(\phi)^2 + \gamma^2}}, \quad j=0,1,$

* $\Delta_j(\phi)=\omega_j(\phi)-\omega_c$,

* and the coupler frequency $\omega_c$ is constant.




The parameters used in the paper are presented below.

<!-- Experiment folder: 20260518_044450_static_4c84263 -->
<!-- Git provenance: commit=ec5eefc12b7f2d6c0ac0a52b515112259108625c (short=ec5eefc), branch=plotting-dev, dirty=true -->

| Effective parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| $\omega_0$ | $x_0$ | 9.746024e+00 |
| $\omega_0$ | $a_1$ | -7.081317e-04 |
| $\omega_0$ | $a_2$ | 1.109688e-03 |
| $\omega_0$ | $a_3$ | 4.732779e-04 |
| $\omega_1$ | $x_0$ | 6.172008e+00 |
| $\omega_1$ | $a_1$ | 2.600343e+00 |
| $\omega_1$ | $a_2$ | -8.207884e-01 |
| $\omega_1$ | $a_3$ | 3.863513e-01 |
| $J$ | $\gamma$ | 1.091991e+00 |
| $J$ | $c_0$ | -3.445362e+05 |
| $J$ | $c_{r_1}$ | 2.101875e+06 |
| $J$ | $c_{r_2}$ | -1.315542e+03 |
| $J$ | $c_\mathrm{prod}$ | 4.011514e+03 |
| $J$ | $c_{r_0,\mathrm{sq}}$ | -3.205669e+06 |
| $J$ | $c_{r_1,\mathrm{sq}}$ | -8.994630e-01 |
| $\zeta$ | $\gamma$ | 1.031616e+00 |
| $\zeta$ | $c_0$ | 1.759212e+04 |
| $\zeta$ | $c_{r_1}$ | -1.065761e+05 |
| $\zeta$ | $c_{r_2}$ | 6.401228e+01 |
| $\zeta$ | $c_\mathrm{prod}$ | -1.938414e+02 |
| $\zeta$ | $c_{r_1,\mathrm{sq}}$ | 1.614140e+05 |
| $\zeta$ | $c_{r_2,\mathrm{sq}}$ | 4.327856e-02 |

## Duffing model
$$
\begin{aligned}
H_{\mathrm{Duff}}(\phi)
={}&
\sum_{j\in\{0,1\}}
\left(
\omega_j(\phi)\, a_j^\dagger a_j
+\frac{\alpha_j}{2} a_j^\dagger a_j^\dagger a_j a_j
\right)
+\omega_c\, a_c^\dagger a_c \\
&\quad
+\sum_{j\in\{0,1\}} g_{j,c}\left(a_j^\dagger a_c+a_c^\dagger a_j\right),
\end{aligned}
$$

<!-- Experiment folder: 20260518_044450_static_4c84263 -->
<!-- Git provenance: commit=ec5eefc12b7f2d6c0ac0a52b515112259108625c (short=ec5eefc), branch=plotting-dev, dirty=true -->

| Duffing parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| $\omega_0$ | c0 | 9.729824e+00 |
| $\omega_1$ | c0 | 6.155486e+00 |
| $\omega_1$ | $cos1$ | 2.643109e+00 |
| $\omega_1$ | $cos2$ | -8.641246e-01 |
| $\omega_1$ | $cos3$ | 4.425637e-01 |
| $\omega_1$ | $cos4$ | -3.709943e-01 |
| $\omega_1$ | $cos5$ | 1.995666e-01 |
| $\alpha_0$ | c0 | -2.610773e-01 |
| $\alpha_1$ | c0 | -4.007571e-01 |
| $\alpha_1$ | cos1 | 8.050788e-02 |
| $\alpha_1$ | cos2 | -7.171780e-02 |
| $\alpha_1$ | cos3 | 4.324207e-02 |
| $w_c$ | c0 | 6.742597e+00 |
| $g_{0,c}$ | c0 | 3.022813e-01 |
| $g_{1,c}$ | c0 | 1.815956e-01 |
| $g_{1,c}$ | cos1 | 7.932721e-02 |
| $g_{1,c}$ | cos2 | -5.528838e-02 |



