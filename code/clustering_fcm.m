%clear; clc; close all;
rng(10)
% Load Data
load('TrueData.mat');
%% Step 1: Define Observation Matrix H
% Define a binary observation matrix H (3x3 identity for full observation, or modify for partial observation)
H = [1, 0, 0; % Only observing x
    0, 1, 0;
    0, 0, 1]; % Only observing z
% Select observed variables based on H
observed_idx = any(H, 2); % Find which variables are observed
% Extract observed data
obs_data = H* [x_obs; y_obs; z_obs];
%obs_data = (obs_data - mean(obs_data, 1)) ./ std(obs_data, 0, 1);
%obs_data = (obs_data-mean(obs_data,2))/std(obs_data,1);
%obs_data = obs_data(:,1:20);
% %% Step 2: Compute ACF for Observed Dimensions
% max_lag = 50; % Define max lag for ACF
% threshold = 0;
% 
% decorrelation_times = NaN(size(H, 1), 1);
% 
% for i = 1:size(H, 1)
% if observed_idx(i)
% acf = autocorr(obs_data(i,:), 'NumLags', max_lag);
% decorrelation_times(i) = find(acf < threshold, 1);
% end
% end
% 
% decorrelation_time = round(mean(decorrelation_times)/2)-1; % Use max of observed decorrelation times
% %decorrelation_time = max(decorrelation_times, [], 'omitnan'); % Use max of observed decorrelation times
% if isempty(decorrelation_time) || isnan(decorrelation_time)
% decorrelation_time = max_lag;
% end
% 
% fprintf('Estimated decorrelation time: %d steps', decorrelation_time);
%% Step 3: Define Window Length and Extract Segments
%window_length = decorrelation_time; % Set window length
window_length = 2;
%window_length = 1; % Set window length
num_segments = size(obs_data,2) - window_length + 1;
segments = zeros(num_segments, window_length * sum(observed_idx));
for i = window_length:num_segments
    segments(i-window_length+1, :) = reshape(obs_data(observed_idx,i-window_length+1:i), 1, []);
end

%% Step 4: Apply Fuzzy C-Means (Soft Clustering)
num_clusters = 2; % Define number of clusters
options = [1.5, 100, 1e-5, 1]; % [Fuzziness, MaxIter, Tolerance, Display]
[centers, U] = fcm(segments, num_clusters, options);
% options = [1.1, 100, 1e-5, 0.3,1]; % lambda_e = 0.1, lambda_b = 1 % [m, max_iter, tol, lambda

% fill the missing weights due to clustering window
U = [U(:,1).*ones(num_clusters,window_length-1),U];

Gamma_t_fcm = U;
save('Clustering_fcm.mat', "Gamma_t_fcm")