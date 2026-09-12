#!/usr/bin/env python3
"""Derives the golden values embedded as expected constants in the C++ tests
test_manifold.cc, test_oracle.cc, and test_interpolate.cc, by calling the
reference implementation (RMO-continuous-cuts, python package
bernoulli_multilevel) directly on fixed inputs.

A "golden value" here is a reference number computed once by a trusted,
independent implementation and then hard-coded as the expected result in a
test; the test passes if the code under test reproduces it (within a small
tolerance). It is "golden" in the sense of being a fixed, trusted baseline,
not something the test itself computes.

Usage:
    python3 derive_golden_values.py /path/to/RMO-continuous-cuts

bernoulli_multilevel's package __init__.py unconditionally imports .problem,
which needs scikit-image; if it is not installed, this script still works
because none of the functions used below need it -- only import the
submodules used here would be needed, but bernoulli_multilevel does not
support that without the package import succeeding first.
"""
import sys

import torch


def main(src_dir: str) -> None:
    sys.path.insert(0, src_dir)

    from bernoulli_multilevel.manifold import exp, lifting
    from bernoulli_multilevel.objective import create_cc_objective
    from bernoulli_multilevel.primitives import (
        r_injection, prolong_bilinear, restrict_adjoint_bilinear,
    )
    from bernoulli_multilevel.operators import BASE_TRIPLES

    torch.set_printoptions(precision=10, sci_mode=False)

    def flat(t):
        return ", ".join(f"{x:.10f}" for x in t.flatten().tolist())

    # --- test_manifold.cc: check_against_reference() ---
    print("# test_manifold.cc")
    phi = torch.tensor([0.3, 0.6, 0.2, 0.8])
    v = torch.tensor([0.1, -0.2, 0.05, 0.3])
    z = exp(phi, v)
    print("exp(phi, v)          =", flat(z), " # -> retract(v, phi)")
    print("lifting(phi, exp(..))=", flat(lifting(phi, z)), " # -> retract_inv(...), should equal v")
    print()

    # --- test_oracle.cc: check_against_reference() ---
    print("# test_oracle.cc")
    rho = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1], [0.2, 0.1, -0.3]])
    coarse = torch.tensor([[0.2, 0.5, 0.7], [0.3, 0.6, 0.1], [0.8, 0.4, 0.9]])
    u0 = coarse.clone().requires_grad_(True)
    # Note: eq. (42) uses epsilon^2 under the square root; create_cc_objective's
    # eps enters unsquared, so pass sqrt(1e-3) from the C++ side to compare like
    # for like (see doc/plan_continuous_cuts.tex, Verification section).
    E0 = create_cc_objective(rho, alpha=0.7, eps=1e-3)(u0)
    E0.backward()
    print(f"E(coarse) = {E0.item():.10f}   # -> value(phi)")
    print("grad(coarse) =", flat(u0.grad), " # -> gradient(phi), via autograd")
    print()

    # --- test_interpolate.cc ---
    print("# test_interpolate.cc")
    fine_v = torch.arange(1., 26.).reshape(5, 5)
    print("r_injection(fine_v) =", flat(r_injection(fine_v)), " # -> to_coarse_mesh")
    print("prolong_bilinear(coarse, (5,5)) =", flat(prolong_bilinear(coarse, (5, 5))), " # -> to_fine_mesh(coarse)")
    print("restrict_adjoint_bilinear(fine_v) =", flat(restrict_adjoint_bilinear(fine_v)), " # -> Tfine(fine_v)")

    w = torch.tensor([[1., 0., 2.], [0., 1., 0.], [2., 0., 1.]])
    lhs = torch.sum(restrict_adjoint_bilinear(fine_v) * w)
    rhs = torch.sum(fine_v * prolong_bilinear(w, (5, 5)))
    print(f"<Tfine(fine_v), w> = {lhs.item():.10f} == <fine_v, to_fine_mesh(w)> = {rhs.item():.10f}  # adjoint identity")

    x_fine = torch.tensor([0.1 + 0.8 * k / 24.0 for k in range(25)]).reshape(5, 5)
    _, R1, P1 = BASE_TRIPLES["Option 1"]
    _, R4, P4 = BASE_TRIPLES["Option 4"]
    print("Option 1 P_op(x_fine, w, psi=coarse) =", flat(P1(x_fine, w, psi=coarse)), " # -> vector_prolongation(...)")
    print("Option 1 == Option 4, P:", torch.allclose(P1(x_fine, w, psi=coarse), P4(x_fine, w, psi=coarse)))
    print("Option 1 == Option 4, R:", torch.allclose(R1(x_fine, fine_v, psi=coarse), R4(x_fine, fine_v, psi=coarse)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} /path/to/RMO-continuous-cuts")
    main(sys.argv[1] + "/src")
