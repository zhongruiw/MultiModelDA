
% Load true data
load('TrueData.mat')
load Clustering.mat
rng(10)

dt = 0.005; % Numerical integration time step

% ETKF parameters
G = [1, 0, 0;
     0, 1, 0;
     0,0,1];  % Observation matrix
Dim_obs = size(G, 1);  % Observation variable dimension
Ens_Total = 200; % Total ensemble size
models = 2;  % Number of models
ini_cov = 2;
Ro = obs_noise^2 * eye(Dim_obs); % Observation noise covariance

% Model parameters
sigma = [10, 20];
beta = [8/3, 5];
rho = [28, 10];

% Initialize ensemble
ensemble = [x_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total);
            y_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total);
            z_truth(1) + sqrt(ini_cov) * randn(1, Ens_Total)];

S_delayed = S_obs; % 先复制原始信号
n = length(S_obs);
for i = 1:n-1
    if S_obs(i) == 1 && S_obs(i+1) == 0
        % 如果不是倒数第一个元素，则将下降边缘延迟一个采样点
        S_delayed(i+1) = 1;  % 延迟效果：原本应为0的位置变为1
        if i+2 <= n
            S_delayed(i+2) = 0; % 后续的采样点置0，确保下降边沿延后
        end
    end
end

S_delayed(76) = 0;
S_delayed(77) = 1;
S_delayed(79) = 0;

Gamma_t = U_new; % Initialize mixture weights
Gamma_t_new = zeros(size(Gamma_t));
% Gamma_1t = S_obs == 1;
% Gamma_2t = 1 - Gamma_1t;
% Gamma_t = [Gamma_1t;Gamma_2t];



% Estimate initial mean and covariance from ensemble
initial_mean = mean(ensemble, 2);
%initial_cov = cov(ensemble');
initial_cov = eye(3);

% Posterior storage
posterior_mean = zeros(3, N/N_gap, models);
posterior_cov = cell(N/N_gap, models);
posterior_weights = zeros(models, N/N_gap);
posterior_GMM_mean = zeros(3, N/N_gap);
posterior_GMM_cov = cell(N/N_gap, 1);

posterior_GMM_mean(:,1) = initial_mean;
posterior_GMM_cov{1} = initial_cov;

% Run MM-EnKF
for ij = 2:N/N_gap
    % Step 1: Allocate ensemble sizes based on prior weights
    %Ens_Num = max(15, round(Gamma_t(:,ij) * Ens_Total));
    Ens_Num = round(Gamma_t(:,ij) * Ens_Total);
    excess = sum(Ens_Num) - Ens_Total;
    if excess > 0
        [~, max_idx] = max(Ens_Num);
        Ens_Num(max_idx) = Ens_Num(max_idx) - excess;
    end
    
    % Step 2: Generate forecasts using respective models
    for j = 2:N_gap
        start_idx = 1;
        for m = 1:models
            end_idx = start_idx + Ens_Num(m) - 1;
            x_new = ensemble(1, start_idx:end_idx) + sigma(m) * (ensemble(2, start_idx:end_idx) - ensemble(1, start_idx:end_idx)) * dt + sigma_x * sqrt(dt) * randn(1, Ens_Num(m));
            y_new = ensemble(2, start_idx:end_idx) + (ensemble(1, start_idx:end_idx) .* (rho(m) - ensemble(3, start_idx:end_idx)) - ensemble(2, start_idx:end_idx)) * dt + sigma_y * sqrt(dt) * randn(1, Ens_Num(m));
            z_new = ensemble(3, start_idx:end_idx) + (ensemble(1, start_idx:end_idx) .* ensemble(2, start_idx:end_idx) - beta(m) * ensemble(3, start_idx:end_idx)) * dt + sigma_z * sqrt(dt) * randn(1, Ens_Num(m));
            
            ensemble(:, start_idx:end_idx) = [x_new; y_new; z_new];
            start_idx = end_idx + 1;
            if start_idx>Ens_Total
                break
            end
        end
    end
    
    % Step 3: EnKF update for each model
    obs_t = G * [x_obs(ij); y_obs(ij); z_obs(ij)];
    likelihoods = zeros(models, 1);
    for m = 1:models
        idx_range = sum(Ens_Num(1:m-1)) + (1:Ens_Num(m));
        if Ens_Num(m)==0
            likelihoods(m) = 0;
        else
            [u_posterior, posterior_mean(:, ij, m)] = EnKF_update(ensemble(:, idx_range), obs_t, G, Ro, Ens_Num(m));
            posterior_cov{ij, m} = ((u_posterior-posterior_mean(:, ij, m)) * (u_posterior-posterior_mean(:, ij, m))') / (Ens_Num(m) - 1);
    
            
            ensemble(:, idx_range) = u_posterior;
            % Compute likelihood
            innovation = obs_t - G * posterior_mean(:, ij, m);
            likelihoods(m) = exp(-0.5 * (innovation' / (G * posterior_cov{ij, m} * G' + Ro) * innovation)) / sqrt(det(2 * pi * (G * posterior_cov{ij, m} * G' + Ro)));
        end
    end
    
    % Step 4: Calculate posterior weights
    if ij == 6;
        flag =1;
    end
    posterior_weights(:, ij) = Gamma_t(:,ij) .* likelihoods;
    posterior_weights(:, ij) = posterior_weights(:, ij) / sum(posterior_weights(:, ij));
    
    % Step 5: Compute posterior GMM mean and covariance
    posterior_GMM_mean(:, ij) = sum(posterior_weights(:, ij)' .* squeeze(posterior_mean(:, ij, :)), 2);
    posterior_GMM_cov{ij} = zeros(3,3);
    for m = 1:models
        if Ens_Num(m) > 0
        diff = posterior_mean(:, ij, m) - posterior_GMM_mean(:, ij);
        posterior_GMM_cov{ij} = posterior_GMM_cov{ij} + posterior_weights(m, ij) * (posterior_cov{ij, m} + (diff * diff'));
        end
    end
end


% Compute RMSE
Truth_all = [x_truth;y_truth;z_truth];
RMSE = sqrt(mean((Truth_all(:,1:N_gap:end) - posterior_GMM_mean).^2, 2));
disp(['RMSE: ', num2str(RMSE')]);

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
figure;
time = dt:N_gap*dt:N*dt;
full_time = dt:dt:N*dt;

for i = 1:3
    subplot('position',[0.04, 0.74-0.22*(i-1),0.93,0.15])
    hold on
    box on
    
    if i == 1
        plot(full_time, x_truth, 'b', 'linewidth', 2);
        title('X variable');
    elseif i == 2
        plot(full_time, y_truth, 'b', 'linewidth', 2);
        title('Y variable');
    else
        plot(full_time, z_truth, 'b', 'linewidth', 2);
        title('Z variable');
    end

    % Confidence interval (shaded region)
    fill([time, fliplr(time)], [ci_upper(i,:), fliplr(ci_lower(i,:))], ...
        'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    % Plot posterior GMM mean
    plot(time, posterior_GMM_mean(i,:), 'r', 'linewidth', 2);
    if i == 1
        legend('Truth','Confidence Interval','Estimated Mean')
    end
    
    % Add pattern correlation text
    x_text = time(end) * 0.8;
    y_text = max(ci_upper(i,:)) * 0.9;
    y_lim = ylim;
    text(x_text, y_text, {['Corr:', num2str(pattern_corr(i), '%.3f')], ['RMSE: ', num2str(RMSE(i), '%.3f')]}, 'FontSize', 14, 'Color', 'k', 'FontWeight', 'bold');
    % Plot settings
    set(gca,'FontSize',16.2)
end

% Regime change plot
subplot('position',[0.04, 0.08,0.93,0.15])
box on
hold on
plot(full_time, S, 'k', 'LineWidth', 1.5);
plot(time,Gamma_t(1,:), 'g--', 'LineWidth', 1.5);
plot(time,posterior_weights(1,:), 'm--', 'LineWidth', 1.5);
ylim([-0.1,1.1]);
title('Regime Change');
set(gca,'FontSize',16.2)
legend({'True Regime Change', 'Prior Weight','Posterior Weight'})
sgtitle('Mixture MM-EnKF with Clustering Prior Weight','fontsize',24)
