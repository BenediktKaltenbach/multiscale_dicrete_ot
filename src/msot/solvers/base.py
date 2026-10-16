class LocalSolver:
    def solve_local(self, rows, columns, c, mu, nu, state=None):
        """OT restricted to N = (rows, columns) with entry costs c.

        Returns (pi, alpha, beta, state): pi a scipy COO matrix, (alpha, beta) local dual optimizers,
        state opaque warm-start data.
        """
        raise NotImplementedError
