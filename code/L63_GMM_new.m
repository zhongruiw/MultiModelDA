%test
% Load true data
load('TrueData.mat')
load Clustering_fcm.mat
rng(10)

dt = 0.005; % Numerical integration time step

% ETKF parameters
G = [1, 0, 0;
     0, 1, 0;
     0, 0, 1];  % Observation matrix
Dim_obs = size(G, 1);  % Observation variable dimension
Ens_Total = 200; % Total ensemble size
models = 2;  % Number of models
ini_cov = 2;
Ro = obs_noise^2 * eye(Dim_obs); % Observation noise covariance

% Model parameters
sigma = [10, 20];
beta = [8/3, 5];
rho = [28, 10];
sigma_x = sqrt(2.000);
sigma_y = 1; %sqrt(12.13)
sigma_z = 1; %sqrt(12.31)

% Initialize ensemble
ensemble = [x_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total);
            y_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total);
            z_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total)];

% Gamma_t = Gamma_t_fcm; % Initialize mixture weights
% % Sometimes adjusting orders is needed since clustering algorthim has no order
% Gamma_t_new = zeros(size(Gamma_t));
% Gamma_t_new(1, :) = Gamma_t(2, :);
% Gamma_t_new(2, :) = Gamma_t(1, :);
% Gamma_t = Gamma_t_new;
Gamma_1t = S_obs == 1;
Gamma_2t = 1 - Gamma_1t;
Gamma_t = [Gamma_1t;Gamma_2t];

% Estimate initial mean and covariance from ensemble
initial_mean = mean(ensemble, 2);
%initial_cov = cov(ensemble');
initial_cov = eye(3);

% Prior storage
prior_mean_m = zeros(3, models);
prior_GMM_mean = zeros(3, N/N_gap);
prior_GMM_MAP = zeros(3, N/N_gap);

% Posterior storage
posterior_mean = zeros(3, N/N_gap, models);
posterior_cov = cell(N/N_gap, models);
posterior_weights = zeros(models, N/N_gap);
posterior_GMM_mean = zeros(3, N/N_gap);
posterior_GMM_cov = cell(N/N_gap, 1);
posterior_GMM_MAP = zeros(3, N/N_gap);

posterior_GMM_mean(:,1) = initial_mean;
posterior_GMM_MAP(:,1) = initial_mean;
posterior_GMM_cov{1} = initial_cov;

% Run MM-EnKF
for ij = 2:N/N_gap
    % Step 1: Allocate ensemble sizes based on prior weights
    Ens_Num = max(15, round(Gamma_t(:,ij) * Ens_Total));
    excess = sum(Ens_Num) - Ens_Total;
    if excess > 0
        [~, max_idx] = max(Ens_Num);
        Ens_Num(max_idx) = Ens_Num(max_idx) - excess;
    end
    
    % Step 2: Generate forecasts using respective models
    start_idx = 1;
    for m = 1:models
        end_idx = start_idx + Ens_Num(m) - 1;
        x_old = ensemble(1, start_idx:end_idx);
        y_old = ensemble(2, start_idx:end_idx);
        z_old = ensemble(3, start_idx:end_idx);
        for j = 2:N_gap
            x_new = x_old + sigma(m) * (y_old - x_old) * dt + sigma_x * sqrt(dt) * randn(1, Ens_Num(m));
            y_new = y_old + (x_old .* (rho(m) - z_old) - y_old) * dt + sigma_y * sqrt(dt) * randn(1, Ens_Num(m));
            z_new = z_old + (x_old .* y_old - beta(m) * z_old) * dt + sigma_z * sqrt(dt) * randn(1, Ens_Num(m));
            x_old = x_new;
            y_old = y_new;
            z_old = z_new;
        end
        ensemble_m = [x_new; y_new; z_new];
        ensemble(:, start_idx:end_idx) = ensemble_m;
        prior_mean_m(:,m) = mean(ensemble_m, 2);
        start_idx = end_idx + 1;
    end
    prior_GMM_mean(:, ij) = sum(Gamma_t(:,ij)' .* prior_mean_m, 2);
   
    % Step 3: EnKF update for each model
    obs_t = G * [x_obs(ij); y_obs(ij); z_obs(ij)];
    likelihoods = zeros(models, 1);
    for m = 1:models
        idx_range = sum(Ens_Num(1:m-1)) + (1:Ens_Num(m));
        [u_posterior, posterior_mean(:, ij, m)] = EnKF_update(ensemble(:, idx_range), obs_t, G, Ro, Ens_Num(m));
        posterior_cov{ij, m} = ((u_posterior-posterior_mean(:, ij, m)) * (u_posterior-posterior_mean(:, ij, m))') / (Ens_Num(m) - 1);

        ensemble(:, idx_range) = u_posterior;
        % Compute likelihood
        innovation = obs_t - G * posterior_mean(:, ij, m);
        likelihoods(m) = exp(-0.5 * (innovation' / (G * posterior_cov{ij, m} * G' + Ro) * innovation)) / sqrt(det(2 * pi * (G * posterior_cov{ij, m} * G' + Ro)));
    end
    
    % Step 4: Calculate posterior weights
    posterior_weights(:, ij) = Gamma_t(:,ij) .* likelihoods;
    posterior_weights(:, ij) = posterior_weights(:, ij) / sum(posterior_weights(:, ij));
    
    % Step 5: Compute posterior GMM mean and covariance
    posterior_GMM_mean(:, ij) = sum(posterior_weights(:, ij)' .* squeeze(posterior_mean(:, ij, :)), 2);
    posterior_GMM_cov{ij} = zeros(3,3);
    for m = 1:models
        diff = posterior_mean(:, ij, m) - posterior_GMM_mean(:, ij);
        posterior_GMM_cov{ij} = posterior_GMM_cov{ij} + posterior_weights(m, ij) * (posterior_cov{ij, m} + (diff * diff'));
    end
end


% Compute RMSE
Truth_all = [x_truth;y_truth;z_truth];
RMSE_prior = sqrt(mean((Truth_all(:,1:N_gap:end) - prior_GMM_mean).^2, 2));
disp(['prior RMSE: ', num2str(RMSE_prior')]);
RMSE = sqrt(mean((Truth_all(:,1:N_gap:end) - posterior_GMM_mean).^2, 2));
disp(['posterior RMSE: ', num2str(RMSE')]);

% Compute pattern correlation
pattern_corr = zeros(3,1);
for i = 1:3
    pattern_corr(i) = corr(Truth_all(i,1:N_gap:end)', posterior_GMM_mean(i,:)');
end
disp(['Pattern Correlation X: ', num2str(pattern_corr(1))]);
disp(['Pattern Correlation Y: ', num2str(pattern_corr(2))]);
disp(['Pattern Correlation Z: ', num2str(pattern_corr(3))]);

% Compute confidence intervals (95% CI)
posterior_std = zeros(3, N/N_gap);
for t = 1:N/N_gap
    posterior_std(:,t) = sqrt(diag(posterior_GMM_cov{t}));
end
ci_upper = posterior_GMM_mean + 2 * posterior_std;
ci_lower = posterior_GMM_mean - 2 * posterior_std;

% Plot results with confidence intervals
N_plot = 80/dt;
figure;
time = dt:N_gap*dt:N_plot*dt;
full_time = dt:dt:N_plot*dt;

for i = 1:3
    subplot('position',[0.04, 0.74-0.22*(i-1),0.93,0.15])
    hold on
    box on
    
    if i == 1
        plot(full_time, x_truth(1:N_plot), 'k', 'linewidth', 2);
        title('X variable');
    elseif i == 2
        plot(full_time, y_truth(1:N_plot), 'k', 'linewidth', 2);
        title('Y variable');
    else
        plot(full_time, z_truth(1:N_plot), 'k', 'linewidth', 2);
        title('Z variable');
    end

    % Confidence interval (shaded region)
    fill([time, fliplr(time)], [ci_upper(i,1:N_plot/N_gap), fliplr(ci_lower(i,1:N_plot/N_gap))], ...
        'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    % Plot posterior GMM mean
    plot(time, prior_GMM_mean(i,1:N_plot/N_gap), 'b', 'linewidth', 2);
    plot(time, posterior_GMM_mean(i,1:N_plot/N_gap), 'r', 'linewidth', 2);
    if i == 1
        legend('Truth','Confidence Interval','Prior Mean','Posterior Mean')
    end
    
    % Add pattern correlation text
    x_text = time(end) * 0.8;
    y_text = max(ci_upper(i,1:N_plot/N_gap)) * 0.9;
    y_lim = ylim;
    text(x_text, y_text, {['Corr:', num2str(pattern_corr(i), '%.3f')], ['prior RMSE: ', num2str(RMSE_prior(i), '%.3f')], ['posterior RMSE: ', num2str(RMSE(i), '%.3f')]}, 'FontSize', 14, 'Color', 'k', 'FontWeight', 'bold');
    % Plot settings
    set(gca,'FontSize',16.2)
end

% Regime change plot
subplot('position',[0.04, 0.08,0.93,0.15])
box on
hold on
plot(full_time, S(1:N_plot), 'k', 'LineWidth', 1.5);
plot(time(:),Gamma_t(1,1:N_plot/N_gap), 'g--', 'LineWidth', 1.5);
plot(time(:),posterior_weights(1,1:N_plot/N_gap), 'm--', 'LineWidth', 1.5);
ylim([-0.1,1.1]);
title('Regime Change');
set(gca,'FontSize',16.2)
legend({'True Regime Change', 'Prior Weight','Posterior Weight'})
sgtitle('Mixture MM-EnKF with True Prior Weight','fontsize',24)
