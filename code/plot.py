import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

################################################### Model #####################################################    
def plot_trajectories(x_truth, y_truth, z_truth, dt, sel0=10000, sel1=20000, interv=10):
    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    gs0 = GridSpec(1, 2, figure=fig)

    # Left column: Individual trajectories
    gs00 = gs0[0].subgridspec(3, 6)
    ax1 = fig.add_subplot(gs00[0, :])
    ax2 = fig.add_subplot(gs00[1, :])
    ax3 = fig.add_subplot(gs00[2, :])

    ax1.plot(xaxis, x_truth[sel0:sel1:interv])
    ax1.set_xlim(xaxis[0], xaxis[-1])
    ax1.set_title('(a) Sample trajectory of x')

    ax2.plot(xaxis, y_truth[sel0:sel1:interv])
    ax2.set_xlim(xaxis[0], xaxis[-1])
    ax2.set_title('(b) Sample trajectory of y')

    ax3.plot(xaxis, z_truth[sel0:sel1:interv])
    ax3.set_xlim(xaxis[0], xaxis[-1])
    ax3.set_title('(c) Sample trajectory of z')
    ax3.set_xlabel('t')

    # Right column: 2D and 3D trajectories
    gs01 = gs0[1].subgridspec(2, 7)
    ax4 = fig.add_subplot(gs01[0, :3])
    ax5 = fig.add_subplot(gs01[0, 4:])
    ax6 = fig.add_subplot(gs01[1, :3])
    ax7 = fig.add_subplot(gs01[1, 4:], projection='3d')

    ax4.plot(x_truth, y_truth, lw=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('(d) 2D trajectory of x and y')

    ax5.plot(y_truth, z_truth, lw=0.5)
    ax5.set_xlabel('y')
    ax5.set_ylabel('z')
    ax5.set_title('(e) 2D trajectory of y and z')

    ax6.plot(z_truth, x_truth, lw=0.5)
    ax6.set_xlabel('z')
    ax6.set_ylabel('x')
    ax6.set_title('(f) 2D trajectory of z and x')

    ax7.plot(x_truth, y_truth, z_truth, lw=0.5)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_title('(g) 3D trajectory of x, y, and z')
    ax7.grid(False)
    plt.tight_layout()


################################################### Clustering #####################################################    
def plot_scatter_weights(score, trueLabels, accuracy_entropy, accuracy_baseline, W_entropy, W_baseline,correct_entropy, correct_baseline, signalDim):
    # Plot entropy regularized results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Entropy Reg. (Acc: {accuracy_entropy*100:.1f}%)")
    plt.grid(True)
    plt.scatter(score[correct_entropy & (trueLabels == 1), 0], score[correct_entropy & (trueLabels == 1), 1], c='b', label='Class 1', s=50)
    plt.scatter(score[correct_entropy & (trueLabels == 2), 0], score[correct_entropy & (trueLabels == 2), 1], c='g', label='Class 2', s=50)
    plt.scatter(score[~correct_entropy, 0], score[~correct_entropy, 1], c='r', marker='x', label='Errors', s=50)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    
    # Plot baseline results
    plt.subplot(1, 2, 2)
    plt.title(f"No Entropy (Acc: {accuracy_baseline*100:.1f}%)")
    plt.grid(True)
    plt.scatter(score[correct_baseline & (trueLabels == 1), 0], score[correct_baseline & (trueLabels == 1), 1], c='b', label='Class 1', s=50)
    plt.scatter(score[correct_baseline & (trueLabels == 2), 0], score[correct_baseline & (trueLabels == 2), 1], c='g', label='Class 2', s=50)
    plt.scatter(score[~correct_baseline, 0], score[~correct_baseline, 1], c='r', marker='x', label='Errors', s=50)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    plt.tight_layout()
    
    # Plot feature weights
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(W_entropy)), W_entropy)
    plt.axvline(x=signalDim - 0.5, color='r', linestyle='--', linewidth=1.5)
    plt.title("Feature Weights with Entropy Regularization")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(W_baseline)), W_baseline)
    plt.axvline(x=signalDim - 0.5, color='r', linestyle='--', linewidth=1.5)
    plt.title("Feature Weights without Entropy")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight")
    plt.grid(True)

    plt.tight_layout()

def plot_histogram_comparison(p_hat, q_hat, bin_edges, title='', var_name='x'):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]) * 0.4

    plt.figure(figsize=(6, 4))
    plt.bar(bin_centers - width/2, p_hat, width=width, color='k', label='Truth', alpha=0.7)
    plt.bar(bin_centers + width/2, q_hat, width=width, color='r', label='Model', alpha=0.7)
    
    plt.xlabel(f'{var_name}')
    plt.ylabel('Probability')
    plt.title(title or f'Histogram comparison for {var_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

def plot_3d_histogram(p, edges, title='3D Histogram', threshold=0.001):
    """
    Plot a 3D voxel representation of a 3D joint histogram using bin edges.
    
    Parameters:
    - p: 3D numpy array of histogram probabilities
    - edges: list of 3 arrays for bin edges in x, y, z
    - title: plot title
    - threshold: minimum density to display
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Only show voxels above threshold
    filled = p > threshold
    values = p[filled] / p.max()

    # Get bin centers
    x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])
    
    # Build 3D grid of centers
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # Extract coordinates where filled
    x_vals = X[filled]
    y_vals = Y[filled]
    z_vals = Z[filled]

    # Prepare color array
    facecolors = plt.cm.gist_gray(values)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Plot each voxel manually as a cube
    size = (edges[0][1] - edges[0][0])  # assume equal-sized bins
    for (x, y, z, color) in zip(x_vals, y_vals, z_vals, facecolors):
        ax.bar3d(x, y, z, size, size, size, color=color, edgecolor='k', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()

def plot_loss(loss_list):
    plt.figure(figsize=(4, 3))
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.title('Loss over Iterations')
    plt.grid(True)
    plt.tight_layout()

def plot_grad_norms(grad_norm_log):
    # Convert from list of PyTrees to arrays
    fuzzypart_norms = np.array([float(g[0]) for g in grad_norm_log])
    center_norms = np.array([float(g[1]) for g in grad_norm_log])
    W_norms = np.array([float(g[2]) for g in grad_norm_log])

    steps = np.arange(len(grad_norm_log))

    plt.figure(figsize=(6, 4))
    plt.plot(steps, fuzzypart_norms, label='fuzzypartmat logits', linewidth=2)
    plt.plot(steps, center_norms, label='centers', linewidth=2)
    plt.plot(steps, W_norms, label='W logits', linewidth=2)

    plt.yscale('log')  # log scale helps visualize small or diverging norms
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (L2)')
    plt.title('Gradient Norms During Optimization')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()


###################################################### DA #######################################################    
def plot_l63_series(dt, sel0, sel1, interv,
                    x_truth, y_truth, z_truth, S,
                    mean, spread,
                    prior_weights, posterior_weights,
                    xlim, warmup=20):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    var_names = ['X', 'Y', 'Z']
    truth_vars = [x_truth, y_truth, z_truth]
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    # Time series plots (X, Y, Z)
    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5, label='Truth')
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5, label='Posterior Mean')
        l3 = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color='r', alpha=0.2, label='Spread')
        ax.set_title(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)

        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        # Correlation and RMSE annotation
        truth_i = truth_vars[i][warmup:]
        mean_i = mean[warmup:, i]
        corr = np.corrcoef(truth_i, mean_i)[0, 1]
        rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2))
        textstr = f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.99, 0.94, textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Regime plot
    ax = axes[3]
    l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
    l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5, label='Prior Weight')
    l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5, label='Posterior Weight')
    lines.extend([l4, l5, l6])
    labels.extend(['True Regime', 'Prior Weight', 'Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )

    # fig.tight_layout()  # Leave space for the global legend    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave enough bottom margin

def plot_gmm_pdf_3vars(means_list, stds_list, weights_list, x_ranges=None, num_points=1000, var_names=['X', 'Y', 'Z']):
    """
    Plot the PDFs of 1D Gaussian mixtures for three variables in subplots.

    Parameters:
    - means_list: list of 3 arrays of means for each variable
    - stds_list: list of 3 arrays of stds for each variable
    - weights_list: list of 3 arrays of weights for each variable
    - x_ranges: optional list of 3 (xmin, xmax) tuples; if None, auto-determined
    - num_points: number of x-points per PDF
    - var_names: list of variable names for labeling
    """
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=(10,3), sharex=False)

    for i in range(3):
        means = np.asarray(means_list[i])
        stds = np.asarray(stds_list[i])
        weights = np.asarray(weights_list)

        if x_ranges is None or x_ranges[i] is None:
            xmin = np.min(means - 4 * stds)
            xmax = np.max(means + 4 * stds)
        else:
            xmin, xmax = x_ranges[i]

        x = np.linspace(xmin, xmax, num_points)
        pdf = np.zeros_like(x)
        for mu, sigma, w in zip(means, stds, weights):
            pdf += w * norm.pdf(x, loc=mu, scale=sigma)

        ax = axes[i]
        ax.plot(x, pdf, 'k')
        for j, mu in enumerate(means):
            ax.axvline(mu, linestyle='--', color='k', alpha=0.5, label=f'$\mu_{j}={mu:.2f}$')
        ax.set_xlabel(f"{var_names[i]}")
        # ax.set_title(f"GMM PDF of {var_names[i]}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    plt.tight_layout()
