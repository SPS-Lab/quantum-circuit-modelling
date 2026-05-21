# Model details

In the effective model

$H_{\mathrm{eff}}(\phi) = \sum_{j\in\{0,1\}} \frac{\tilde{\omega}_j(\phi)}{2} Z_j
+J(\phi)\bigl(X_1X_0+Y_1Y_0\bigr)
+\frac{\zeta(\phi)}{4} Z_1 Z_0,$

the dressed qubit frequencies are modelled as harmonics

$\tilde\omega_j = 2\pi x_{j,0} + \sum_{k=1}^3 a_{j,k}\cos(2\pi k\phi), \quad j=0,1.$

$\omega_c$ is a constant.

$J$ and $\zeta$ as

$r_j = \frac{1}{\sqrt{\Delta_j^2 + \gamma^2}}, \quad j=0,1,$



$O = c_{O,0}, \quad O=J,\zeta$


The parameters used in the paper are presented below.

<!-- Experiment folder: 20260518_044450_static_4c84263 -->
<!-- Git provenance: commit=ec5eefc12b7f2d6c0ac0a52b515112259108625c (short=ec5eefc), branch=plotting-dev, dirty=true -->

| Effective parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| w0 | x0 | 9.746024e+00 |
| w0 | a1 | -7.081317e-04 |
| w0 | a2 | 1.109688e-03 |
| w0 | a3 | 4.732779e-04 |
| w1 | x0 | 6.172008e+00 |
| w1 | a1 | 2.600343e+00 |
| w1 | a2 | -8.207884e-01 |
| w1 | a3 | 3.863513e-01 |
| J | gamma | 1.091991e+00 |
| J | c0 | -3.445362e+05 |
| J | c_r1 | 2.101875e+06 |
| J | c_r2 | -1.315542e+03 |
| J | c_prod | 4.011514e+03 |
| J | c_r1_sq | -3.205669e+06 |
| J | c_r2_sq | -8.994630e-01 |
| zeta | gamma | 1.031616e+00 |
| zeta | c0 | 1.759212e+04 |
| zeta | c_r1 | -1.065761e+05 |
| zeta | c_r2 | 6.401228e+01 |
| zeta | c_prod | -1.938414e+02 |
| zeta | c_r1_sq | 1.614140e+05 |
| zeta | c_r2_sq | 4.327856e-02 |

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

