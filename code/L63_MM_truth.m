% Generate the truth signal of the noisy L63 model with regime switching
rng(10)

% Time parameters
T = 1e3; % total time length
dt = 0.005; % numerical integration time step
dt_obs = 0.5; % observation time step
obs_noise = 2*sqrt(2);
N = round(T/dt); % total numerical integration steps
N_gap = round(dt_obs/dt); % observational gap

% Lorenz-63 parameters
sigma1 = 10;
sigma2 = 20;
beta1 = 8/3;
beta2 = 5;
rho1 = 28;
rho2 = 10;  % Alternate regime value

sigma_x = sqrt(2.000);
sigma_y = 1;
sigma_z = 1;

% Regime switching (Markov process)
lambda_12 = 0.2; % Transition from rho1 -> rho2
lambda_21 = 0.3; % Transition from rho2 -> rho1
S = zeros(1, N); % Store regime state (0 or 1)
S(1) = 1; % Start with rho1

% Initialize states
x_truth = zeros(1, N);
y_truth = zeros(1, N);
z_truth = zeros(1, N);
x_truth(1) = 1.508870;
y_truth(1) = -1.531271;
z_truth(1) = 25.46091;

% Generate the true signal with regime-switching
for i = 2:N
    % Stochastic regime switching
    if S(i-1) == 1
        if rand < lambda_12 * dt
            S(i) = 0;
        else
            S(i) = 1;
        end
    else
        if rand < lambda_21 * dt
            S(i) = 1;
        else
            S(i) = 0;
        end
    end
    rho = rho1 * (S(i) == 1) + rho2 * (S(i) == 0);
    sigma = sigma1 * (S(i) == 1) + sigma2 * (S(i) == 0);
    beta = beta1 * (S(i) == 1) + beta2 * (S(i) == 0);

    % Stochastic Lorenz system
    x_truth(i) = x_truth(i-1) + sigma * (y_truth(i-1) - x_truth(i-1)) * dt + sigma_x * sqrt(dt) * randn;
    y_truth(i) = y_truth(i-1) + (x_truth(i-1) * (rho - z_truth(i-1)) - y_truth(i-1)) * dt + sigma_y * sqrt(dt) * randn;
    z_truth(i) = z_truth(i-1) + (x_truth(i-1) * y_truth(i-1) - beta * z_truth(i-1)) * dt + sigma_z * sqrt(dt) * randn;
end

% Generate observations
x_obs = x_truth(1:N_gap:end) + randn(1, N/N_gap) * obs_noise;
y_obs = y_truth(1:N_gap:end) + randn(1, N/N_gap) * obs_noise;
z_obs = z_truth(1:N_gap:end) + randn(1, N/N_gap) * obs_noise;

S_obs = S(1:N_gap:end);

% Save for ETKF
save('TrueData.mat', 'x_truth', 'y_truth', 'z_truth', 'S', 'x_obs', 'y_obs', 'z_obs', 'N', 'N_gap', 'dt', 'dt_obs','S_obs', 'obs_noise', 'sigma_x', 'sigma_y', 'sigma_z');
