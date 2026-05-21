# Model details

## Effective model

The effective model is

$$H_{\mathrm{eff}}(\phi) = \sum_{j\in\{0,1\}} \frac{\tilde{\omega}_j(\phi)}{2} Z_j
+J(\phi)\bigl(X_1X_0+Y_1Y_0\bigr)
+\frac{\zeta(\phi)}{4} Z_1 Z_0.$$

The dressed qubit frequencies are modelled as harmonics

$$\tilde\omega_j(\phi) = 2\pi x_{j,0} + \sum_{k=1}^3 a_{j,k}\cos(2\pi k\phi), \quad j=0,1.$$

$J$ and $\zeta$ is modelled with one coefficient set

$$\\{\gamma~,~c_{0}~,~c_{r_0}~,~c_{r_1}~,~c_{\mathrm{prod}}~,~c_{{r_0},\mathrm{sq}}~,~c_{{r_1},\mathrm{sq}}\\}$$

each as

$$O(\phi) = c_{0} + c_{r_1}~r_0(\phi) + c_{r_2}~r_1(\phi) + c_{\mathrm{prod}}~(r_1(\phi)~r_2(\phi)) + c_{r_0,\mathrm{sq}}~r_0(\phi)^2 + c_{r_1,\mathrm{sq}}~r_0(\phi)^2\quad O=J,\zeta$$

where

* the residuals $r_j(\phi) = \frac{1}{\sqrt{\Delta_j(\phi)^2 + \gamma^2}}, \quad j=0,1,$

* $\Delta_j(\phi)=\omega_j(\phi)-\omega_c$,

* and the coupler frequency $\omega_c$ is constant.




The parameters used in the paper are presented below.

<!-- Experiment folder: 20260518_044450_static_4c84263 -->
<!-- Git provenance: commit=ec5eefc12b7f2d6c0ac0a52b515112259108625c (short=ec5eefc), branch=plotting-dev, dirty=true -->

| Effective parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| $w_0$ | $x_0$ | 9.746024e+00 |
| $w_0$ | $a_1$ | -7.081317e-04 |
| $w_0$ | $a_2$ | 1.109688e-03 |
| $w_0$ | $a_3$ | 4.732779e-04 |
| $w_1$ | $x_0$ | 6.172008e+00 |
| $w_1$ | $a_1$ | 2.600343e+00 |
| $w_1$ | $a_2$ | -8.207884e-01 |
| $w_1$ | $a_3$ | 3.863513e-01 |
| $J$ | $\gamma$ | 1.091991e+00 |
| $J$ | $c_0$ | -3.445362e+05 |
| $J$ | $c_r1$ | 2.101875e+06 |
| $J$ | $c_r2$ | -1.315542e+03 |
| $J$ | $c_\mathrm{prod}$ | 4.011514e+03 |
| $J$ | $c_{{r_1},sq}$ | -3.205669e+06 |
| $J$ | $c_r2_sq$ | -8.994630e-01 |
| $\zeta$ | $\gamma$ | 1.031616e+00 |
| $\zeta$ | $c0$ | 1.759212e+04 |
| $\zeta$ | $c_r1$ | -1.065761e+05 |
| $\zeta$ | $c_r2$ | 6.401228e+01 |
| $\zeta$ | $c_prod$ | -1.938414e+02 |
| $\zeta$ | $c_r1_sq$ | 1.614140e+05 |
| $\zeta$ | $c_r2_sq$ | 4.327856e-02 |

## Symbolic Duffing fitted coefficients
<!-- Experiment folder: 20260518_044450_static_4c84263 -->
<!-- Git provenance: commit=ec5eefc12b7f2d6c0ac0a52b515112259108625c (short=ec5eefc), branch=plotting-dev, dirty=true -->

| Duffing parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| w0 | c0 | 9.729824e+00 |
| w1 | c0 | 6.155486e+00 |
| w1 | cos1 | 2.643109e+00 |
| w1 | cos2 | -8.641246e-01 |
| w1 | cos3 | 4.425637e-01 |
| w1 | cos4 | -3.709943e-01 |
| w1 | cos5 | 1.995666e-01 |
| alpha0 | c0 | -2.610773e-01 |
| alpha1 | c0 | -4.007571e-01 |
| alpha1 | cos1 | 8.050788e-02 |
| alpha1 | cos2 | -7.171780e-02 |
| alpha1 | cos3 | 4.324207e-02 |
| wc | c0 | 6.742597e+00 |
| g0c | c0 | 3.022813e-01 |
| g1c | c0 | 1.815956e-01 |
| g1c | cos1 | 7.932721e-02 |
| g1c | cos2 | -5.528838e-02 |

## Duffing model calibration details

An initial physics informed guess of $\omega_j$ and $\alpha_j$ for j=0,1 is made using scqubits for single Transmons for each flux point, since $\alpha$ can not be taken from the $4\times4$ projection $E_{\text{circ},\text{proj}}$.

