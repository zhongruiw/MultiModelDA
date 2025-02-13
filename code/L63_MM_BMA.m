% Load true data
load('TrueData.mat')
rng(10)

% ETKF parameters
Dim_obs = 3;
Ens_Num = 50;
ini_cov = 2;
Ro = obs_noise^2 * eye(Dim_obs);
r = 0;

% Initial ensemble setup
x_sample1 = x_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
y_sample1 = y_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
z_sample1 = z_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
x_sample2 = x_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
y_sample2 = y_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
z_sample2 = z_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
x_sample3 = x_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
y_sample3 = y_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);
z_sample3 = z_truth(1) + sqrt(ini_cov) * randn(1, Ens_Num);

% Posterior storage
posterior_weighted = zeros(3, N/N_gap, Ens_Num);
posterior_store = zeros(3, N/N_gap);
posterior_weighted_compare = zeros(3, N/N_gap, Ens_Num);
posterior_store_compare = zeros(3, N/N_gap);

% Initialize with true state
posterior_weighted(:, 1, :) = repmat([x_truth(1); y_truth(1); z_truth(1)], 1, Ens_Num);
posterior_store(:,1) = [x_truth(1); y_truth(1); z_truth(1)];

posterior_weighted_compare(:, 1, :) = repmat([x_truth(1); y_truth(1); z_truth(1)], 1, Ens_Num);
posterior_store_compare(:,1) = [x_truth(1); y_truth(1); z_truth(1)];

% Pseudo-clustering setup
epsilon_t = 0.1 + (0.3 - 0.1) * rand(1, N/N_gap);
Gamma_1t = (S(1:N_gap:end) == 1) .* (1 - epsilon_t) + (S(1:N_gap:end) == 0) .* epsilon_t;
Gamma_2t = 1 - Gamma_1t;

% Observation matrix
G = eye(3); % Full observation of x, y, z

% Run ETKF with BMA propagation
for ij = 2:N/N_gap
    for i = 2:N_gap
    % Get the previous step weighted posterior ensemble
        % Model 1 evolution (rho1)
        x_sample_1_new = x_sample1 + sigma1 * (y_sample1 - x_sample1) * dt + 2*sigma_x * sqrt(dt) * randn(1, Ens_Num);
        y_sample_1_new = y_sample1 + (x_sample1 .* (rho1 - z_sample1) - y_sample1) * dt + 2*sigma_y * sqrt(dt) * randn(1, Ens_Num);
        z_sample_1_new = z_sample1 + (x_sample1 .* y_sample1 - beta1 * z_sample1) * dt + sigma_z * sqrt(dt) * randn(1, Ens_Num);
        
        % Model 2 evolution (rho2)
        x_sample_2_new = x_sample2 + sigma2 * (y_sample2 - x_sample2) * dt + 0.5*sigma_x * sqrt(dt) * randn(1, Ens_Num);
        y_sample_2_new = y_sample2 + (x_sample2 .* (rho2 - z_sample2) - y_sample2) * dt + sigma_y * sqrt(dt) * randn(1, Ens_Num);
        z_sample_2_new = z_sample2 + (x_sample2 .* y_sample2 - beta2 * z_sample2) * dt + 4*sigma_z* sqrt(dt) * randn(1, Ens_Num);


        % Model 2 evolution (rho3)
        x_sample_3_new = x_sample3 + sigma2 * (y_sample3 - x_sample3) * dt + sigma_x * sqrt(dt) * randn(1, Ens_Num);
        y_sample_3_new = y_sample3 + (x_sample3 .* (rho2 - z_sample3) - y_sample3) * dt + sigma_y * sqrt(dt) * randn(1, Ens_Num);
        z_sample_3_new = z_sample3 + (x_sample3 .* y_sample3 - beta2 * z_sample3) * dt + sigma_z* sqrt(dt) * randn(1, Ens_Num);


        x_sample1 = x_sample_1_new;
        y_sample1 = y_sample_1_new;
        z_sample1 = z_sample_1_new;

        x_sample2 = x_sample_2_new;
        y_sample2 = y_sample_2_new;
        z_sample2 = z_sample_2_new;


        x_sample3 = x_sample_3_new;
        y_sample3 = y_sample_3_new;
        z_sample3 = z_sample_3_new;
    end
    % Compute individual priors
    u_prior_1 = [x_sample1; y_sample1; z_sample1];
    u_prior_2 = [x_sample2; y_sample2; z_sample2];
    u_prior_3 = [x_sample3; y_sample3; z_sample3];



    obs_t = [x_obs(ij); y_obs(ij); z_obs(ij)]; % Current observation

    % Compute individual posteriors
    [u_posterior_1,u_posterior_mean_1] = EnKF_update(u_prior_1, obs_t, G, Ro, Ens_Num);
    [u_posterior_2,u_posterior_mean_2] = EnKF_update(u_prior_2, obs_t, G, Ro, Ens_Num);
    [u_posterior_3,u_posterior_mean_3] = EnKF_update(u_prior_3, obs_t, G, Ro, Ens_Num);

    % Compute **Weighted Posterior Ensemble** using BMA
    u_posterior_w = Gamma_1t(ij) .* u_posterior_1 + Gamma_2t(ij) .* u_posterior_2;
    u_posterior_mean = Gamma_1t(ij) .* u_posterior_mean_1 + Gamma_2t(ij) .* u_posterior_mean_2;
    %u_posterior_w = u_posterior_1;
    %u_posterior_mean = u_posterior_mean_1;

    % Store posterior ensemble for next step propagation
    posterior_weighted(:, ij, :) = u_posterior_w;
    posterior_store(:, ij) = u_posterior_mean;

    posterior_weighted_compare(:, ij, :) = u_posterior_3;
    posterior_store_compare(:, ij, :) = u_posterior_mean_3;

    x_sample1 = u_posterior_w(1,:);
    y_sample1 = u_posterior_w(2,:);
    z_sample1 = u_posterior_w(3,:);

    x_sample2 = u_posterior_w(1,:);
    y_sample2 = u_posterior_w(2,:);
    z_sample2 = u_posterior_w(3,:);

    x_sample3 = u_posterior_3(1,:);
    y_sample3 = u_posterior_3(2,:);
    z_sample3 = u_posterior_3(3,:);
end

% Save posterior weighted ensemble
save('PosteriorWeighted.mat', 'posterior_weighted','posterior_store','posterior_store_compare', 'N', 'N_gap', 'dt', 'dt_obs');

% Compute individual posterior ensembles via ETKF
