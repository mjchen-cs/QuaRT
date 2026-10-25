"""Discrete-modulated (QPSK) CV-QKD key-rate evaluation, with and without a
quantum-scissors (QS) noiseless linear amplifier at the receiver.

Two independent routes are implemented:
  * ``analytic_cm``   - closed-form covariance matrix of the QPSK entanglement-
                        based (EB) state through a thermal-loss channel
                        (Leverrier & Grangier, PRL 2009);
  * ``fock_*``        - a truncated Fock-space simulation of the very same EB
                        state, the thermal-loss channel (beam splitter with a
                        thermal environment), and the ideal quantum scissors.
The Fock route without QS reproduces the analytic covariance matrix and is used
as a numerical cross-check (see ``self_test``).

Security: asymptotic regime, collective attacks, reverse reconciliation, Holevo
bound upper-bounded through Gaussian optimality of the symmetrised covariance
matrix (as in Ghalaii et al., IEEE JSTQE 2020).  Shot-noise units (SNU):
vacuum quadrature variance = 1, x = a + a^dagger.
"""
import numpy as np
from math import factorial, comb, sqrt

LOG2 = np.log(2.0)


# ---------------------------------------------------------------- helpers ---
def G(nu):
    """von Neumann entropy (bits) of a thermal mode with symplectic eigenvalue nu."""
    nu = np.maximum(nu, 1.0 + 1e-12)
    a, b = (nu + 1) / 2, (nu - 1) / 2
    out = a * np.log2(a)
    out = out - np.where(b > 1e-15, b * np.log2(np.maximum(b, 1e-300)), 0.0)
    return out


def holevo_rr(VA, VB, C):
    """Holevo bound chi_BE (reverse reconciliation, homodyne at Bob) from the
    symmetric CM [[VA I, C Z],[C Z, VB I]] (Gaussian optimality)."""
    Delta = VA ** 2 + VB ** 2 - 2 * C ** 2
    D = VA * VB - C ** 2
    disc = np.sqrt(np.maximum(Delta ** 2 - 4 * D ** 2, 0.0))
    nu1 = np.sqrt(np.maximum((Delta + disc) / 2, 1.0))
    nu2 = np.sqrt(np.maximum((Delta - disc) / 2, 1.0))
    nu3 = np.sqrt(max(VA * (VA - C ** 2 / VB), 1.0))
    return float(G(nu1) + G(nu2) - G(nu3))


def lambdas(alpha):
    a2 = alpha ** 2
    e = np.exp(-a2)
    return np.array([0.5 * e * (np.cosh(a2) + np.cos(a2)),
                     0.5 * e * (np.sinh(a2) + np.sin(a2)),
                     0.5 * e * (np.cosh(a2) - np.cos(a2)),
                     0.5 * e * (np.sinh(a2) - np.sin(a2))])


def qpsk_Z(alpha):
    lam = lambdas(alpha)
    # ordering of Leverrier-Grangier: lam_{k-1}^{3/2} lam_k^{-1/2}
    s = sum(lam[(k - 1) % 4] ** 1.5 / np.sqrt(lam[k]) for k in range(4))
    return 2 * alpha ** 2 * s


def analytic_cm(alpha, T, eps):
    V = 1 + 2 * alpha ** 2
    VB = T * (V - 1) + 1 + T * eps
    C = np.sqrt(T) * qpsk_Z(alpha)
    return V, VB, C


# --------------------------------------------------- mutual information ---
_XG = np.linspace(-14, 14, 4001)
_DX = _XG[1] - _XG[0]


def _hermite_wavefunctions(nmax, x):
    """Real Fock wavefunctions psi_n(x) for x = a + a^dagger (vacuum var. 1)."""
    psi = np.zeros((nmax + 1, x.size))
    psi[0] = (2 * np.pi) ** -0.25 * np.exp(-x ** 2 / 4)
    if nmax >= 1:
        psi[1] = x * psi[0]
    for n in range(2, nmax + 1):
        psi[n] = (x * psi[n - 1] - np.sqrt(n - 1) * psi[n - 2]) / np.sqrt(n)
    return psi


def mutual_info_from_densities(pk):
    """I(K;X) in bits for equiprobable K with conditional densities pk[k, x]."""
    pk = np.maximum(pk, 1e-300)
    px = pk.mean(0)
    return float(np.mean(np.sum(pk * np.log2(pk / px), axis=1) * _DX))


def mi_gaussian_qpsk(alpha, T, eps):
    """Exact I(K;X_B) for QPSK through a thermal-loss channel (no QS)."""
    var = 1 + T * eps
    mus = [2 * np.sqrt(T) * alpha * np.cos((2 * k + 1) * np.pi / 4) for k in range(4)]
    pk = np.array([np.exp(-(_XG - m) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var) for m in mus])
    return mutual_info_from_densities(pk)


# ------------------------------------------------------------ Fock route ---
def coherent(alpha, nc):
    n = np.arange(nc)
    return np.exp(-abs(alpha) ** 2 / 2) * alpha ** n / np.sqrt([float(factorial(int(k))) for k in n])


def eb_state(alpha, nc):
    """|Psi>_AB = sum_k sqrt(lambda_k) |phi_k>_A |phi_k>_B, as an nc x nc matrix."""
    lam = lambdas(alpha)
    Psi = np.zeros((nc, nc))
    for k in range(4):
        phi = np.zeros(nc)
        for n in range(k, nc, 4):
            m = (n - k) // 4
            phi[n] = (-1) ** m * alpha ** n / np.sqrt(float(factorial(n)))
        phi *= np.exp(-alpha ** 2 / 2) / np.sqrt(lam[k])
        Psi += np.sqrt(lam[k]) * np.outer(phi, phi)
    return Psi


def bs_tensor(T, nin, nenv, nout):
    """U[b, e, n, m] = <b,e| U_BS |n,m> with b^dag -> t b^dag + r e^dag,
    e^dag -> -r b^dag + t e^dag."""
    t, r = np.sqrt(T), np.sqrt(1 - T)
    U = np.zeros((nout, nout, nin, nenv))
    for n in range(nin):
        for m in range(nenv):
            norm = 1.0 / np.sqrt(float(factorial(n) * factorial(m)))
            for j in range(n + 1):
                for k in range(m + 1):
                    b = j + k
                    e = n - j + m - k
                    if b >= nout or e >= nout:
                        continue
                    c = comb(n, j) * t ** j * r ** (n - j) * comb(m, k) * (-r) ** k * t ** (m - k)
                    U[b, e, n, m] += c * norm * np.sqrt(float(factorial(b) * factorial(e)))
    return U


def thermal_probs(nbar, nenv):
    if nbar < 1e-14:
        p = np.zeros(nenv); p[0] = 1; return p
    m = np.arange(nenv)
    p = nbar ** m / (1 + nbar) ** (m + 1)
    return p / p.sum()


def channel_on_B(Psi, T, eps, nenv=10, nout=None):
    """Apply the thermal-loss channel (T, excess noise eps referred to the
    input) on mode B of the pure EB state. Returns rho_AB as [a,b,a',b']."""
    nc = Psi.shape[0]
    nout = nout or nc
    nbar = T * eps / (2 * (1 - T)) if T < 1 else 0.0
    pm = thermal_probs(nbar, nenv)
    U = bs_tensor(T, nc, nenv, nout + nenv)
    nA = nc
    rho = np.zeros((nA, nout, nA, nout))
    for m in range(nenv):
        if pm[m] < 1e-14:
            continue
        # v[a, b, e] = sum_n Psi[a, n] U[b, e, n, m]
        v = np.einsum('an,ben->abe', Psi, U[:, :, :, m])[:, :nout, :]
        rho += pm[m] * np.einsum('abe,cde->abcd', v, v)
    return rho


def qs_kraus(eta, nout):
    """Ideal quantum scissors (photon-number-resolving detectors, one click):
    M = sqrt(eta)|0><0| + sqrt(1-eta)|1><1| ; gain g = sqrt((1-eta)/eta)."""
    M = np.zeros((2, nout))
    M[0, 0] = np.sqrt(eta)
    M[1, 1] = np.sqrt(1 - eta)
    return M


def quad_ops(n):
    a = np.diag(np.sqrt(np.arange(1, n)), 1)
    x = a + a.T
    p = 1j * (a.T - a)
    return x, p


def cm_from_rho(rho):
    """Symmetrised CM entries (VA, VB, C) of a two-mode density matrix [a,b,a',b']."""
    # pad by two Fock levels so that x^2, p^2 are exact on the support of rho
    a0, b0 = rho.shape[0], rho.shape[1]
    nA, nB = a0 + 2, b0 + 2
    rp = np.zeros((nA, nB, nA, nB), dtype=rho.dtype)
    rp[:a0, :b0, :a0, :b0] = rho
    R = rp.reshape(nA * nB, nA * nB)
    xA, pA = quad_ops(nA)
    xB, pB = quad_ops(nB)
    IA, IB = np.eye(nA), np.eye(nB)
    ev = lambda O: float(np.real(np.trace(R @ O)))
    VA = 0.5 * (ev(np.kron(xA @ xA, IB)) + ev(np.kron(pA @ pA, IB)))
    VB = 0.5 * (ev(np.kron(IA, xB @ xB)) + ev(np.kron(IA, pB @ pB)))
    C = 0.5 * (ev(np.kron(xA, xB)) - ev(np.kron(pA, pB)))
    return VA, VB, C


def fock_rate(alpha, T, eps, beta=0.95, eta=None, nc=14, nenv=10):
    """Key rate (bits/pulse) through Fock simulation. eta=None -> no QS."""
    nout = nc if eta is None else 8
    Psi = eb_state(alpha, nc)
    rho = channel_on_B(Psi, T, eps, nenv=nenv, nout=nout)
    if eta is None:
        P = 1.0
        rho_c = rho
    else:
        M = qs_kraus(eta, nout)
        rho_c = np.einsum('ib,abcd,jd->aicj', M, rho, M)
        nA = rho_c.shape[0]
        P = float(np.trace(rho_c.reshape(nA * 2, nA * 2)))
        rho_c = rho_c / P
    VA, VB, C = cm_from_rho(rho_c)
    chi = holevo_rr(VA, VB, C)
    # exact I(K;X_B) in the prepare-and-measure picture
    psi = _hermite_wavefunctions(rho_c.shape[1] - 1, _XG)
    pk = []
    for k in range(4):
        ak = alpha * np.exp(1j * (2 * k + 1) * np.pi / 4)
        st = coherent(ak, nc)
        # Bob's conditional state: channel acting on a single coherent state
        rB = _bob_state_coherent(st, T, eps, nenv, nout)
        if eta is not None:
            M = qs_kraus(eta, nout)
            rB = M @ rB @ M.T
            rB = rB / np.trace(rB)
        dens = np.real(np.einsum('ix,ij,jx->x', psi, rB, psi))
        pk.append(dens)
    I = mutual_info_from_densities(np.array(pk))
    K = P * (beta * I - chi)
    return dict(K=K, P=P, I=I, chi=chi, VA=VA, VB=VB, C=C)


def _bob_state_coherent(st, T, eps, nenv, nout):
    """Bob's density matrix (complex) for a coherent input |st> through the channel."""
    nc = st.size
    nbar = T * eps / (2 * (1 - T)) if T < 1 else 0.0
    pm = thermal_probs(nbar, nenv)
    U = bs_tensor(T, nc, nenv, nout + nenv)
    rho = np.zeros((nout, nout), dtype=complex)
    for m in range(nenv):
        if pm[m] < 1e-14:
            continue
        v = np.einsum('n,ben->be', st, U[:, :, :, m])[:nout, :]
        rho += pm[m] * v @ v.conj().T
    return rho


def analytic_rate(alpha, T, eps, beta=0.95):
    VA, VB, C = analytic_cm(alpha, T, eps)
    chi = holevo_rr(VA, VB, C)
    I = mi_gaussian_qpsk(alpha, T, eps)
    return dict(K=beta * I - chi, P=1.0, I=I, chi=chi, VA=VA, VB=VB, C=C)


def transmittance(L_km, att_db_per_km=0.2, extra_db=0.0):
    return 10 ** (-(att_db_per_km * L_km + extra_db) / 10)


def optimise_rate(L_km, eps, beta=0.95, qs=False, alphas=None, etas=None, **kw):
    """Maximise the key rate over the modulation amplitude (and QS eta)."""
    T = transmittance(L_km, **kw)
    alphas = np.linspace(0.3, 1.3, 11) if alphas is None else alphas
    best = dict(K=-np.inf)
    if not qs:
        for a in alphas:
            r = analytic_rate(a, T, eps, beta)
            if r['K'] > best['K']:
                best = dict(r, alpha=a)
    else:
        etas = np.array([0.5, 0.35, 0.2, 0.1, 0.05, 0.02]) if etas is None else etas
        for a in alphas:
            for e in etas:
                r = fock_rate(a, T, eps, beta, eta=e)
                if r['K'] > best['K']:
                    best = dict(r, alpha=a, eta=e)
    best['T'] = T
    best['L'] = L_km
    return best


def self_test():
    """Fock route (no QS) must reproduce the analytic CM."""
    for (a, T, e) in [(0.6, 0.5, 0.02), (1.0, 0.2, 0.01)]:
        VA, VB, C = analytic_cm(a, T, e)
        r = fock_rate(a, T, e, nc=18, nenv=12)
        print(f"alpha={a} T={T} eps={e}:  analytic (VA,VB,C)=({VA:.4f},{VB:.4f},{C:.4f})  "
              f"fock=({r['VA']:.4f},{r['VB']:.4f},{r['C']:.4f})  K_fock={r['K']:.4e}  "
              f"K_analytic={analytic_rate(a, T, e)['K']:.4e}")


if __name__ == '__main__':
    self_test()
