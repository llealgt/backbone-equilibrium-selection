"""
Three constructed sequential games with EXACT analytic Nash families.

Motivation
----------
The original study ran three games but only one of them could test the
hypothesis: the matrix games tokenize to a single token (maxT=1), which makes
the attention block a linear map, and they have only N=2 information sets, so a
shared-parameter network faces no cross-infoset generalisation pressure. That
pressure is the sole channel by which an architectural bias could reach
equilibrium selection.

Adding a *natural* second game is harder than it looks: Leduc has no closed-form
Nash family, and computing one numerically would introduce approximate ground
truth into a study where every other quantity is exact. We therefore CONSTRUCT
sequential games whose Nash sets are exactly characterisable, by duplicating an
action that is payoff-identical and (where relevant) unobservable. Any split of
probability mass between exact duplicates is an equilibrium, so the Nash set
acquires extra free coordinates that we can write down in closed form.

The three games are a designed contrast, not independent games. They share
Kuhn's core and vary ONE property: where the free equilibrium parameters live.

    kuhn_dupcall : P0's own LEAF decision      (low reach, own-side)
    kuhn_dupp1   : the OPPONENT's leaf decision (low reach, opponent-side)
    kuhn_dupbet  : P0's ROOT decision, unobservable to P1 (HIGH reach, own-side)

Reach-weighting is central to the fitted-target objective, so "where the free
parameters sit" is exactly the axis a shared-parameter network should be
sensitive to. Sharing a game core is a limitation (stated in the paper); it is
also what makes this a controlled comparison rather than a grab-bag.

Every family below is verified numerically to be exactly Nash (NashConv <= 1e-12)
across its coordinate range by `validate_all()`.
"""
import numpy as np
import efg

TERMINAL = {(0, 0), (0, 1, 0), (0, 1, 1), (1, 0), (1, 1)}


def _kuhn_util(c0, c1, h):
    """Standard Kuhn payoffs on a CANONICAL history."""
    win = 1 if c0 > c1 else -1
    if h == (0, 0):    return win
    if h == (1, 0):    return 1           # P0 bet, P1 folded
    if h == (1, 1):    return 2 * win     # bet-call
    if h == (0, 1, 0): return -1          # P0 checked, P1 bet, P0 folded
    if h == (0, 1, 1): return 2 * win     # check-bet-call
    raise ValueError(h)


# ===========================================================================
# G1: duplicate P0's CALL after check-bet  (own-side, leaf, low reach)
# ===========================================================================
class KuhnDupCall(efg.Game):
    """P0's response to a bet has three actions {fold, call_a, call_b}; call_a
    and call_b are payoff-identical and terminal, so any split of the call mass
    is an equilibrium. Free coordinates: (eta, s_Q, t_K)."""
    DECK = [0, 1, 2]
    name = "kuhn_dupcall"
    coord_name = "bluff freq P(bet|J)"
    coord_range = (0.0, 1 / 3)
    maxent = None

    def _canon(self, h):
        return tuple(1 if (len(h) == 3 and i == 2 and a == 2) else a
                     for i, a in enumerate(h))

    def roots(self):
        d = [(c0, c1) for c0 in self.DECK for c1 in self.DECK if c0 != c1]
        return [(1.0 / len(d), (c0, c1, ())) for c0, c1 in d]

    def is_terminal(self, node):
        return self._canon(node[2]) in TERMINAL

    def util(self, node):
        c0, c1, h = node
        return _kuhn_util(c0, c1, self._canon(h))

    def player(self, node):
        return 0 if len(node[2]) in (0, 2) else 1

    def infoset(self, node):
        c0, c1, h = node
        p = self.player(node)
        return (p, c0 if p == 0 else c1, h)

    def actions(self, node):
        return [0, 1, 2] if node[2] == (0, 1) else [0, 1]

    def actions_for_infoset(self, I):
        return [0, 1, 2] if I[2] == (0, 1) else [0, 1]

    def next_node(self, node, a):
        c0, c1, h = node
        return (c0, c1, h + (a,))

    def selection_coord(self, sigma):
        return sigma[(0, 0, ())][1]

    def aux_coords(self, sigma):
        q = sigma[(0, 1, (0, 1))]; k = sigma[(0, 2, (0, 1))]
        sq = q[1] / max(q[1] + q[2], 1e-12)
        tk = k[1] / max(k[1] + k[2], 1e-12)
        return dict(split_Q=float(sq), split_K=float(tk))


def dupcall_family(eta, sQ=0.5, tK=0.5):
    s = {}
    s[(0, 0, ())] = np.array([1 - eta, eta])
    s[(0, 1, ())] = np.array([1.0, 0.0])
    s[(0, 2, ())] = np.array([1 - 3 * eta, 3 * eta])
    s[(0, 0, (0, 1))] = np.array([1.0, 0.0, 0.0])                     # J folds
    cq = 1 / 3 + eta
    s[(0, 1, (0, 1))] = np.array([1 - cq, cq * sQ, cq * (1 - sQ)])    # Q calls cq
    s[(0, 2, (0, 1))] = np.array([0.0, tK, 1 - tK])                   # K always calls
    s[(1, 0, (0,))] = np.array([2 / 3, 1 / 3])
    s[(1, 1, (0,))] = np.array([1.0, 0.0])
    s[(1, 2, (0,))] = np.array([0.0, 1.0])
    s[(1, 0, (1,))] = np.array([1.0, 0.0])
    s[(1, 1, (1,))] = np.array([2 / 3, 1 / 3])
    s[(1, 2, (1,))] = np.array([0.0, 1.0])
    return s


# ===========================================================================
# G2: duplicate P1's CALL facing a bet  (opponent-side, leaf, low reach)
# ===========================================================================
class KuhnDupP1(efg.Game):
    """P1's response to a bet has three actions {fold, call_a, call_b}. The free
    coordinates now live in the OPPONENT's information sets, while the primary
    coordinate (P0's bluff frequency) is unchanged."""
    DECK = [0, 1, 2]
    name = "kuhn_dupp1"
    coord_name = "bluff freq P(bet|J)"
    coord_range = (0.0, 1 / 3)
    maxent = None

    def _canon(self, h):
        return tuple(1 if (len(h) == 2 and i == 1 and a == 2 and h[0] == 1) else a
                     for i, a in enumerate(h))

    def roots(self):
        d = [(c0, c1) for c0 in self.DECK for c1 in self.DECK if c0 != c1]
        return [(1.0 / len(d), (c0, c1, ())) for c0, c1 in d]

    def is_terminal(self, node):
        return self._canon(node[2]) in TERMINAL

    def util(self, node):
        c0, c1, h = node
        return _kuhn_util(c0, c1, self._canon(h))

    def player(self, node):
        return 0 if len(node[2]) in (0, 2) else 1

    def infoset(self, node):
        c0, c1, h = node
        p = self.player(node)
        return (p, c0 if p == 0 else c1, h)

    def actions(self, node):
        return [0, 1, 2] if node[2] == (1,) else [0, 1]

    def actions_for_infoset(self, I):
        return [0, 1, 2] if I[2] == (1,) else [0, 1]

    def next_node(self, node, a):
        c0, c1, h = node
        return (c0, c1, h + (a,))

    def selection_coord(self, sigma):
        return sigma[(0, 0, ())][1]

    def aux_coords(self, sigma):
        q = sigma[(1, 1, (1,))]; k = sigma[(1, 2, (1,))]
        return dict(split_Q=float(q[1] / max(q[1] + q[2], 1e-12)),
                    split_K=float(k[1] / max(k[1] + k[2], 1e-12)))


def dupp1_family(eta, sQ=0.5, tK=0.5):
    s = {}
    s[(0, 0, ())] = np.array([1 - eta, eta])
    s[(0, 1, ())] = np.array([1.0, 0.0])
    s[(0, 2, ())] = np.array([1 - 3 * eta, 3 * eta])
    s[(0, 0, (0, 1))] = np.array([1.0, 0.0])
    s[(0, 1, (0, 1))] = np.array([2 / 3 - eta, 1 / 3 + eta])
    s[(0, 2, (0, 1))] = np.array([0.0, 1.0])
    s[(1, 0, (0,))] = np.array([2 / 3, 1 / 3])
    s[(1, 1, (0,))] = np.array([1.0, 0.0])
    s[(1, 2, (0,))] = np.array([0.0, 1.0])
    s[(1, 0, (1,))] = np.array([1.0, 0.0, 0.0])                    # J folds vs bet
    s[(1, 1, (1,))] = np.array([2 / 3, sQ / 3, (1 - sQ) / 3])      # Q calls 1/3
    s[(1, 2, (1,))] = np.array([0.0, tK, 1 - tK])                  # K always calls
    return s


# ===========================================================================
# G3: duplicate P0's BET at the ROOT, unobservable to P1 (high reach)
# ===========================================================================
class KuhnDupBet(efg.Game):
    """P0's opening has three actions {check, bet_a, bet_b}. The two bets are
    payoff-identical AND indistinguishable to P1, so P1's information sets
    canonicalise them. The free coordinates therefore sit at the ROOT, where
    information-set reach is highest -- the reach-weighted fitting objective
    should be most sensitive here."""
    DECK = [0, 1, 2]
    name = "kuhn_dupbet"
    coord_name = "bluff freq P(bet|J)"
    coord_range = (0.0, 1 / 3)
    maxent = None

    def _canon(self, h):
        if not h:
            return h
        return (min(h[0], 1),) + tuple(h[1:])

    def roots(self):
        d = [(c0, c1) for c0 in self.DECK for c1 in self.DECK if c0 != c1]
        return [(1.0 / len(d), (c0, c1, ())) for c0, c1 in d]

    def is_terminal(self, node):
        return self._canon(node[2]) in TERMINAL

    def util(self, node):
        c0, c1, h = node
        return _kuhn_util(c0, c1, self._canon(h))

    def player(self, node):
        return 0 if len(node[2]) in (0, 2) else 1

    def infoset(self, node):
        c0, c1, h = node
        p = self.player(node)
        # P0 knows which button it pressed; P1 cannot distinguish the two bets.
        return (0, c0, h) if p == 0 else (1, c1, self._canon(h))

    def actions(self, node):
        return [0, 1, 2] if node[2] == () else [0, 1]

    def actions_for_infoset(self, I):
        return [0, 1, 2] if I[2] == () else [0, 1]

    def next_node(self, node, a):
        c0, c1, h = node
        return (c0, c1, h + (a,))

    def selection_coord(self, sigma):
        r = sigma[(0, 0, ())]
        return float(r[1] + r[2])                 # TOTAL bluff mass

    def aux_coords(self, sigma):
        j = sigma[(0, 0, ())]; k = sigma[(0, 2, ())]
        return dict(split_J=float(j[1] / max(j[1] + j[2], 1e-12)),
                    split_K=float(k[1] / max(k[1] + k[2], 1e-12)))


def dupbet_family(eta, uJ=0.5, uK=0.5):
    s = {}
    s[(0, 0, ())] = np.array([1 - eta, eta * uJ, eta * (1 - uJ)])
    s[(0, 1, ())] = np.array([1.0, 0.0, 0.0])
    b = 3 * eta
    s[(0, 2, ())] = np.array([1 - b, b * uK, b * (1 - uK)])
    s[(0, 0, (0, 1))] = np.array([1.0, 0.0])
    s[(0, 1, (0, 1))] = np.array([2 / 3 - eta, 1 / 3 + eta])
    s[(0, 2, (0, 1))] = np.array([0.0, 1.0])
    s[(1, 0, (0,))] = np.array([2 / 3, 1 / 3])
    s[(1, 1, (0,))] = np.array([1.0, 0.0])
    s[(1, 2, (0,))] = np.array([0.0, 1.0])
    s[(1, 0, (1,))] = np.array([1.0, 0.0])
    s[(1, 1, (1,))] = np.array([2 / 3, 1 / 3])
    s[(1, 2, (1,))] = np.array([0.0, 1.0])
    return s


GAMES2 = {"kuhn_dupcall": (KuhnDupCall(), dupcall_family),
          "kuhn_dupp1":   (KuhnDupP1(),   dupp1_family),
          "kuhn_dupbet":  (KuhnDupBet(),  dupbet_family)}


# ===========================================================================
def validate_all(n_eta=13, n_split=5, verbose=True):
    """Verify each constructed family is EXACTLY Nash across its coordinate
    range, and that the tokenization is non-degenerate."""
    import nbn
    out = {}
    for name, (g, fam) in GAMES2.items():
        worst = 0.0
        for eta in np.linspace(0.0, 1 / 3, n_eta):
            for a in np.linspace(0.02, 0.98, n_split):
                for b in np.linspace(0.02, 0.98, n_split):
                    worst = max(worst, abs(efg.nash_conv_dp(g, fam(eta, a, b))))
        F = nbn.build_features(g)
        out[name] = dict(max_nashconv=float(worst), N=int(F["N"]),
                         maxT=int(F["maxT"]), maxA=int(F["maxA"]))
        if verbose:
            print("%-14s max NashConv over family = %.3e | N=%d maxT=%d maxA=%d | %s"
                  % (name, worst, F["N"], F["maxT"], F["maxA"],
                     "OK" if worst < 1e-9 and F["maxT"] > 1 and F["N"] > 2 else "FAIL"))
    return out
