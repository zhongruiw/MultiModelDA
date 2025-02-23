
% function [u_posterior,u_mean_posterior] = ETKF_update(u_prior, obs, G, Ro, Ens_Num, r)
% u_mean_prior = mean(u_prior, 2);
% U = u_prior - u_mean_prior * ones(1, Ens_Num);
% V = G * U;
% 
% % ETKF update
% J = (Ens_Num - 1) / (1 + r) * eye(Ens_Num) + V' / Ro * V;
% J = (J + J') / 2; % Ensure numerical stability
% [X, Gamma] = eig(J);
% x = J \ V' / Ro * ( obs - G * u_mean_prior );
% 
% % Compute posterior mean
% u_mean_posterior = u_mean_prior + U * x;
% 
% % Compute transformation matrix
% T = sqrt(Ens_Num-1) * X * Gamma^(-1/2) * X';
% U_perturb_posterior = U * T;
% u_posterior = u_mean_posterior * ones(1, Ens_Num) + U_perturb_posterior;
% end
% function [u_posterior,u_mean_posterior] = EnKF_update(u_prior, obs, G, Ro, Ens_Num)
% % Compute the ensemble mean and covariance
% u_mean_prior = mean(u_prior, 2); % Prior mean
% U = u_prior - u_mean_prior * ones(1, Ens_Num); % Ensemble perturbations
% %U = u_prior - u_mean_prior;
% P_prior = (U * U') / (Ens_Num - 1); % Prior covariance
% 
% % Compute Kalman gain
% K = P_prior * G' / (G * P_prior * G' + Ro);
% 
% % Generate perturbed observations
% obs_perturbed = obs + sqrtm(Ro) * randn(size(obs, 1), Ens_Num);
% 
% % Update each ensemble member
% u_posterior = u_prior + K * (obs_perturbed - G * u_prior);
% u_mean_posterior = mean(u_posterior, 2);
% end
function [u_posterior, u_mean_posterior] = EnKF_update(u_prior, obs, G, Ro, Ens_Num)
    % Ensure Ens_Num > 1 to prevent division by zero
    if Ens_Num < 2
        warning('Ens_Num is too small, skipping update.');
        u_posterior = u_prior;
        u_mean_posterior = mean(u_posterior, 2);
        return;
    end
    % Compute the ensemble mean and perturbations
    u_mean_prior = mean(u_prior, 2); % Prior mean
    U = u_prior - repmat(u_mean_prior, 1, Ens_Num); % Perturbations
    % Compute prior covariance matrix with regularization
    epsilon = 1e-6; % Small value to prevent singular matrices
    P_prior = (U * U') / (Ens_Num - 1) + epsilon * eye(size(U,1));
    % Compute Kalman gain
    P_yy = G * P_prior * G' + Ro + epsilon * eye(size(Ro)); % Regularized for inversion safety
    K = P_prior * G' / P_yy;
    % Generate perturbed observations
    obs_perturbed = obs + sqrtm(Ro) * randn(size(obs, 1), Ens_Num);
    % Update each ensemble member
    u_posterior = u_prior + K * (obs_perturbed - G * u_prior);
    u_mean_posterior = mean(u_posterior, 2);
end