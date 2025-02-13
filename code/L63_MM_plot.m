% Load Data
load('TrueData.mat')
load('PosteriorWeighted.mat')

% Compute the final weighted ensemble mean
posterior_mean = posterior_store;

% Compute RMSE
Truth_all = [x_truth(1:N_gap:end); y_truth(1:N_gap:end); z_truth(1:N_gap:end)];
RMSE = sqrt(mean((Truth_all - posterior_mean).^2, 2));
disp(['RMSE: ', num2str(RMSE')]);

% Plot results
figure;
for i = 1:3
    subplot(4,1,i)
    hold on
    box on
    if i == 1
        plot(dt:dt:N*dt, x_truth, 'b', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store(i,:), 'r', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store_compare(i,:), 'g--', 'linewidth', 2);
        %plot(dt:N_gap*dt:N*dt, x_obs, 'ro', 'linewidth',2);
        title('X variable');
        legend('Truth','MM-EnKF(BMA)','EnKF(Regime 2)')
        set(gca,'FontSize',12)
    elseif i == 2
        plot(dt:dt:N*dt, y_truth, 'b', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store(i,:), 'r', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store_compare(i,:), 'g--', 'linewidth', 2);
        %plot(dt:N_gap*dt:N*dt, y_obs, 'ro', 'linewidth',2);
        title('Y variable');
        set(gca,'FontSize',12)
    else
        plot(dt:dt:N*dt, z_truth, 'b', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store(i,:), 'r', 'linewidth', 2);
        plot(dt:N_gap*dt:N*dt, posterior_store_compare(i,:), 'g--', 'linewidth', 2);
        %plot(dt:N_gap*dt:N*dt, z_obs, 'ro', 'linewidth',2);
        title('Z variable');
        set(gca,'FontSize',12)
    end
end
subplot(4,1,4)
plot(dt:dt:N*dt,S,'k','LineWidth',1.5)
ylim([-0.1,1.1])
title('Regime Change');
set(gca,'FontSize',12)