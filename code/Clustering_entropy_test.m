%% High-Dimensional, Low Sample Size Testbed
% Construct a dataset with 40 samples and 100 features, where only 5 features carry informative signal and the remaining 95 are noise.
rng(42); % Set fixed random seed

% Set number of samples and features
nSamples = 30;
nFeatures = 100;
signalDim = 5;               % Informative signal dimension
noiseDim = nFeatures - signalDim;  % Noise dimension

%% Construct the signal part (two classes)
% Set the means and covariance for the two classes in the signal dimensions
mu1 = 1 * ones(1, signalDim); 
mu2 = -2 * ones(1, signalDim);
sigma_signal = 2 * eye(signalDim); % Smaller variance for the signal
clusterSize = nSamples / 2; % Each cluster has 20 samples

% Generate signal data for the two classes separately
X_signal1 = mvnrnd(mu1, sigma_signal, clusterSize);
X_signal2 = mvnrnd(mu2, 1.5 * sigma_signal, clusterSize);
X_signal = [X_signal1; X_signal2];

%% Construct the noise part
% Noise data: Gaussian noise with mean 0 and larger variance
sigma_noise = 5;
X_noise = sigma_noise * randn(nSamples, noiseDim);

%% Concatenate to form the final dataset
dataHD = [X_signal, X_noise];  % Size: 40 x 100
dataHD = (dataHD - mean(dataHD, 2)) ./ std(dataHD);
trueLabels = [ones(clusterSize, 1); 2 * ones(clusterSize, 1)];  % True class labels

%% Set parameters for fcm_modified
% options = [m, max_iter, tol, lambda_e]
% options = [1.8, 300, 1e-6, 2.5]; 
options = [1.8, 300, 1e-6, 2.5];
% In this example, each feature is treated individually, i.e., D = nFeatures
D = nFeatures;

%% Run FCM with entropy regularization (λ_e > 0)
[centers_entropy, fuzzypartmat_entropy, objfcn_entropy, W_entropy] = ...
    fcm_modified(dataHD, 2, options, D);

%% Baseline: Run FCM without entropy regularization (λ_e = 0)
options_baseline = options;
options_baseline(4) = 0;  % Set λ_e = 0, equivalent to no entropy regularization
[centers_baseline, fuzzypartmat_baseline, objfcn_baseline, W_baseline] = ...
    fcm_modified(dataHD, 2, options_baseline, D);

%% Evaluate clustering results (accounting for possible label swapping)
[~, assignedLabels_entropy] = max(fuzzypartmat_entropy, [], 1);
assignedLabels_entropy = assignedLabels_entropy';
acc1_entropy = sum(assignedLabels_entropy == trueLabels) / nSamples;
acc2_entropy = sum(assignedLabels_entropy == (3 - trueLabels)) / nSamples;
accuracy_entropy = max(acc1_entropy, acc2_entropy);

[~, assignedLabels_baseline] = max(fuzzypartmat_baseline, [], 1);
assignedLabels_baseline = assignedLabels_baseline';
acc1_baseline = sum(assignedLabels_baseline == trueLabels) / nSamples;
acc2_baseline = sum(assignedLabels_baseline == (3 - trueLabels)) / nSamples;
accuracy_baseline = max(acc1_baseline, acc2_baseline);

fprintf('Clustering Accuracy with Entropy: %.2f%%\n', accuracy_entropy * 100);
fprintf('Clustering Accuracy Baseline (No Entropy): %.2f%%\n', accuracy_baseline * 100);

%% Adjust label mapping to align assignedLabels with trueLabels
% For the entropy regularized results
if sum(assignedLabels_entropy == trueLabels) < sum(assignedLabels_entropy == (3 - trueLabels))
    assignedLabels_entropy = 3 - assignedLabels_entropy;
end
correctEntropy = (assignedLabels_entropy == trueLabels);
accuracy_entropy = sum(correctEntropy) / nSamples;

% For the baseline (no entropy regularization) results
if sum(assignedLabels_baseline == trueLabels) < sum(assignedLabels_baseline == (3 - trueLabels))
    assignedLabels_baseline = 3 - assignedLabels_baseline;
end
correctBaseline = (assignedLabels_baseline == trueLabels);
accuracy_baseline = sum(correctBaseline) / nSamples;

%% Scatter plot after PCA: correctly classified points are colored according to class, misclassified points are marked with red crosses
[coeff, score, ~] = pca(dataHD);

% For entropy regularized results:
% Adjust label mapping to align assignedLabels with trueLabels
if sum(assignedLabels_entropy == trueLabels) < sum(assignedLabels_entropy == (3 - trueLabels))
    assignedLabels_entropy = 3 - assignedLabels_entropy;
end
correctEntropy = (assignedLabels_entropy == trueLabels);
accuracy_entropy = sum(correctEntropy) / nSamples;

% Get indices for correctly classified points per class
correct1_entropy = find(correctEntropy & (trueLabels == 1));
correct2_entropy = find(correctEntropy & (trueLabels == 2));
error_entropy = find(~correctEntropy);

figure;
subplot(1, 2, 1);
hold on;
box on;
% Correctly classified points: Class 1 as blue circles, Class 2 as green circles
scatter(score(correct1_entropy, 1), score(correct1_entropy, 2), 50, 'b', 'filled');
scatter(score(correct2_entropy, 1), score(correct2_entropy, 2), 50, 'g', 'filled');
% Misclassified points: marked with red crosses
scatter(score(error_entropy, 1), score(error_entropy, 2), 50, 'r', 'x');
title({sprintf('Entropy Reg. (Acc: %.1f%%)', accuracy_entropy * 100), ...
       'Class 1: Blue, Class 2: Green, Errors: Red X'});
xlabel('PC1'); ylabel('PC2');
grid on;
set(gca, 'Fontsize', 14);

% For baseline (no entropy regularization) results:
if sum(assignedLabels_baseline == trueLabels) < sum(assignedLabels_baseline == (3 - trueLabels))
    assignedLabels_baseline = 3 - assignedLabels_baseline;
end
correctBaseline = (assignedLabels_baseline == trueLabels);
accuracy_baseline = sum(correctBaseline) / nSamples;

correct1_baseline = find(correctBaseline & (trueLabels == 1));
correct2_baseline = find(correctBaseline & (trueLabels == 2));
error_baseline = find(~correctBaseline);

subplot(1, 2, 2);
hold on;
box on;
scatter(score(correct1_baseline, 1), score(correct1_baseline, 2), 50, 'b', 'filled');
scatter(score(correct2_baseline, 1), score(correct2_baseline, 2), 50, 'g', 'filled');
scatter(score(error_baseline, 1), score(error_baseline, 2), 50, 'r', 'x');
title({sprintf('No Entropy (Acc: %.1f%%)', accuracy_baseline * 100), ...
       'Class 1: Blue, Class 2: Green, Errors: Red X'});
xlabel('PC1'); ylabel('PC2');
grid on;
set(gca, 'Fontsize', 14);

%% Plot comparison of feature weights
figure;
bar(W_entropy);
hold on;
plot([5.5, 5.5], ylim, 'r--', 'LineWidth', 1.5);
title('Feature Weights with Entropy Regularization');
xlabel('Feature Index'); ylabel('Weight');
set(gca, 'FontSize', 14);

% Uncomment below to plot baseline feature weights (if needed)
% figure;
% subplot(1, 2, 2);
% bar(W_baseline);
% title('Feature Weights Baseline (No Entropy)');
% xlabel('Feature Index'); ylabel('Weight');
