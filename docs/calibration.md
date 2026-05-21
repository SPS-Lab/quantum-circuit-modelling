# Full model descriptions

## Effective fitted coefficients
<!-- Experiment folder: 20260517_220950_static_6d83c23_dirty -->
<!-- Git provenance: commit=6d83c2381409bccbf08dba8fed428cadf6268831 (short=6d83c23), branch=main, dirty=true -->

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
<!-- Experiment folder: 20260517_220950_static_6d83c23_dirty -->
<!-- Git provenance: commit=6d83c2381409bccbf08dba8fed428cadf6268831 (short=6d83c23), branch=main, dirty=true -->

| Duffing parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| w0 | c0 | 9.729298e+00 |
| w1 | c0 | 6.187922e+00 |
| w1 | cos1 | 2.574884e+00 |
| w1 | cos2 | -9.367395e-01 |
| w1 | cos3 | 4.824440e-01 |
| w1 | cos4 | -2.611571e-01 |
| w1 | cos5 | 3.387265e-01 |
| alpha0 | c0 | -2.325522e-01 |
| alpha1 | c0 | -4.140643e-01 |
| alpha1 | cos1 | 1.077214e-01 |
| alpha1 | cos2 | -1.038704e-01 |
| alpha1 | cos3 | 7.519371e-02 |
| alpha1 | cos4 | -4.976438e-02 |
| alpha1 | cos5 | 4.464384e-02 |
| g0c | c0 | 2.949338e-01 |
| g1c | c0 | 1.982043e-01 |
| g1c | cos1 | 4.206806e-02 |
| g1c | cos2 | -8.518904e-02 |
| g1c | cos3 | 2.631005e-02 |
| g1c | cos4 | 2.256285e-02 |
| g1c | cos5 | 1.736769e-02 |

## Duffing model calibration details

An initial physics informed guess of $\omega_j$ and $\alpha_j$ for j=0,1 is made using scqubits for single Transmons for each flux point, since $\alpha$ can not be taken from the $4\times4$ projection $E_{\text{circ},\text{proj}}$.

