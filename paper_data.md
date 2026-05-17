# Full model descriptions

## Effective model coefficients
[comment]: <> (Experiment folder: `20260513_154331_static_3f96e1a_dirty`)

[comment]: <> (Git provenance `commit=3f96e1aa47acd50ccb7b9e7a3ca02e60e4d184d4`, `short=3f96e1a`, `branch=main`, `dirty=true`)


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
| J | c_r1_sq | -3.205668e+06 |
| J | c_r2_sq | -8.994630e-01 |
| zeta | gamma | 1.031616e+00 |
| zeta | c0 | 1.759212e+04 |
| zeta | c_r1 | -1.065761e+05 |
| zeta | c_r2 | 6.401228e+01 |
| zeta | c_prod | -1.938414e+02 |
| zeta | c_r1_sq | 1.614140e+05 |
| zeta | c_r2_sq | 4.327857e-02 |

## Duffing model coefficients

| Duffing parameter | Coefficient | Value (GHz) |
| --- | --- | ---: |
| w0 | c0 | 9.729298e+00 |
| w1 | c0 | 6.187921e+00 |
| w1 | cos1 | 2.574883e+00 |
| w1 | cos2 | -9.367379e-01 |
| w1 | cos3 | 4.824475e-01 |
| w1 | cos4 | -2.611572e-01 |
| w1 | cos5 | 3.387232e-01 |
| alpha0 | c0 | -2.325539e-01 |
| alpha1 | c0 | -4.140714e-01 |
| alpha1 | cos1 | 1.077387e-01 |
| alpha1 | cos2 | -1.038942e-01 |
| alpha1 | cos3 | 7.521243e-02 |
| alpha1 | cos4 | -4.979063e-02 |
| alpha1 | cos5 | 4.467040e-02 |
| g0c | c0 | 2.949344e-01 |
| g1c | c0 | 1.982029e-01 |
| g1c | cos1 | 4.206755e-02 |
| g1c | cos2 | -8.518665e-02 |
| g1c | cos3 | 2.631006e-02 |
| g1c | cos4 | 2.256151e-02 |
| g1c | cos5 | 1.736806e-02 |

## Duffing model calibration details

An initial physics informed guess of $\omega_j$ and $\alpha_j$ for j=0,1 is made using scqubits for single Transmons for each flux point, since $\alpha$ can not be taken from the $4\times4$ projection $E_{\text{circ},\text{proj}}$.

