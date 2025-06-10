import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softmax
import optax
from functools import partial


class FCMEntropy:
    def __init__(self, num_clusters, m=2.0, lambda_e=1e-2, lr=1e-2, num_steps=500, seed=0):
        self.num_clusters = num_clusters
        self.m = m
        self.lambda_e = lambda_e
        self.lr = lr
        self.num_steps = num_steps
        self.seed = seed

        # These will be set after training
        self.centers = None
        self.weights = None
        self.params = None

    # ========== JAX-based loss ========== #
    @staticmethod
    @partial(jax.jit, static_argnames=["m", "lambda_e"])
    def loss(params, data, m, lambda_e):
        fuzzypartmat_logits, centers, W_logits = params
        M, N = data.shape
        num_clusters = centers.shape[0]

        fuzzypartmat = softmax(fuzzypartmat_logits, axis=1)
        W = softmax(W_logits)
        fuzzypartmat_m = fuzzypartmat ** m

        def cluster_loss(k):
            diff = data - centers[k]
            weighted_sq = jnp.sum(W * diff**2, axis=1)
            return jnp.sum(fuzzypartmat_m[:, k] * weighted_sq)

        total_loss = jnp.sum(jax.vmap(cluster_loss)(jnp.arange(num_clusters))) / M
        entropy_term = jnp.sum(W * jnp.log(jnp.maximum(W, jnp.finfo(float).eps)))
        total_loss += lambda_e * entropy_term

        return total_loss

    def init_params(self, data):
        M, N = data.shape
        key = jax.random.PRNGKey(self.seed)
        key1, key2, key3 = jax.random.split(key, 3)

        fuzzypartmat_logits = jax.random.normal(key1, (M, self.num_clusters))
        centers = jax.random.uniform(key2, (self.num_clusters, N),
                                     minval=jnp.min(data), maxval=jnp.max(data))
        W_logits = jax.random.normal(key3, (N,))
        
        return (fuzzypartmat_logits, centers, W_logits)

    def fit(self, data, optimizer='gradient_descent', init_params=None, tol=1e-6):
        if optimizer == 'gradient_descent':
            return self._fit_jax(data, init_params)
        elif optimizer == 'iterative':
            return self._fit_iterative(np.array(data), tol)
        else:
            raise ValueError(f"Unknown method: {optimizer}")
    
    def _fit_jax(self, data, init_params=None):
        if init_params is None:
            params = self.init_params(data)
        else:
            params = init_params
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(self.loss)(params, data, self.m, self.lambda_e)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            grad_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
            return params, opt_state, loss, grad_norms

        loss_history = []
        grad_norm_log = []
        for i in range(self.num_steps):
            params, opt_state, loss, grad_norms = step(params, opt_state)
            loss_history.append(loss)
            grad_norm_log.append(grad_norms)

        fuzzypartmat_logits, centers, W_logits = params
        fuzzypartmat = softmax(fuzzypartmat_logits, axis=1)
        W = softmax(W_logits)

        self.centers = centers
        self.weights = W
        self.params = params

        return {
            'centers': centers,
            'weights': W,
            'membership': fuzzypartmat,
            'loss_history': jnp.array(loss_history),
            'grad_norm_log': grad_norm_log,
        }

    def _fit_iterative(self, data, tol):
        m, max_iter, lambda_e = self.m, self.num_steps, self.lambda_e
        M, N = data.shape
        K = self.num_clusters

        centers = np.random.rand(K, N) * (data.max(axis=0) - data.min(axis=0)) + data.min(axis=0)
        W = np.ones(N) / N
        loss_history = np.zeros(max_iter)

        for iter in range(max_iter):
            centers_prev, W_prev = centers.copy(), W.copy()

            # Update membership values
            dist = np.zeros((M, K))
            for k in range(K):
                diff = data - centers[k]
                dist[:, k] = np.sum(W * (diff ** 2), axis=1)
            dist = np.maximum(dist, np.finfo(float).eps)
            tmp = dist ** (-1 / (m - 1))
            fuzzypartmat = tmp / np.sum(tmp, axis=1, keepdims=True)

            # Update cluster centers
            fuzzypartmat_m = fuzzypartmat ** m
            denom = np.sum(fuzzypartmat_m, axis=0)
            centers = (fuzzypartmat_m.T @ data) / denom[:, None]

            # Update feature weights W 
            a = np.zeros(N)
            for d in range(N):
                diff = data[:, d][:, None] - centers[:, d][None, :]
                a[d] = np.sum(fuzzypartmat_m * (diff ** 2))

            if lambda_e == 0:
                W = np.ones(N) / N
            else:
                exp_terms = np.exp(-a / lambda_e)
                if np.all(exp_terms == 0) or np.any(np.isnan(exp_terms)):
                    print(f"W calculation failed at iteration {iter + 1}")
                    W = W_prev
                else:
                    W = exp_terms / np.sum(exp_terms)

            # Compute the loss function
            loss = 0
            for k in range(K):
                diff = data - centers[k]
                loss += np.sum(fuzzypartmat_m[:, k] * np.sum(W * (diff ** 2), axis=1))
            loss = loss / M + lambda_e * np.sum(W * np.log(np.maximum(W, np.finfo(float).eps)))
            loss_history[iter] = loss

            if iter > 0 and np.linalg.norm(centers - centers_prev) < tol and np.linalg.norm(W - W_prev) < tol:
                loss_history = loss_history[:iter + 1]
                break

        self.centers = centers
        self.weights = W

        return {
            'centers': self.centers,
            'weights': self.weights,
            'membership': fuzzypartmat,
            'loss_history': loss_history,
        }

    def predict(self, data, hard=False):
        """
        Predict fuzzy membership of new data based on learned centers and weights.

        Parameters:
        - data: shape (M, N)
        - hard: if True, return hard cluster assignment (argmax)

        Returns:
        - membership matrix: shape (M, K) or hard labels (M,)
        """
        assert self.centers is not None and self.weights is not None, "Model not trained yet."
        W = self.weights

        def compute_dist(x):
            diffs = self.centers - x  # shape (K, N)
            dists = jnp.sum(W * diffs**2, axis=1)  # shape (K,)
            return dists

        dist_all = jax.vmap(compute_dist)(data)  # shape (M, K)
        dist_all = np.maximum(dist_all, np.finfo(float).eps)
        tmp = dist_all ** (-1 / (self.m - 1))
        fuzzypartmat = (tmp / np.sum(tmp, axis=1)[:,None])  # shape (M, K)

        if hard:
            return jnp.argmax(fuzzypartmat, axis=1)
        return fuzzypartmat
