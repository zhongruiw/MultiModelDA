% function [centers, fuzzypartmat, objfcn] = fcm_modified(data, num_clusters, options)
% if nargin < 2
%     num_clusters = 2; 
% end
% if nargin < 3
%     options = [2, 100, 1e-5]; 
% end
% 
% s = options(1); % Fuzziness exponent
% max_iter = options(2); % Maximum number of iterations
% tol = options(3); % Convergence threshold
% 
% [M, N] = size(data);
% data = double(data);
% 
% rng('default');
% centers = rand(num_clusters, N) .* range(data) + min(data);
% objfcn = zeros(max_iter, 1);
% 
% for iter = 1:max_iter
%     centers_prev = centers;
% 
%     dist = zeros(M, num_clusters);
%     for k = 1:num_clusters
%         diff = data - centers(k, :);
%         dist(:, k) = sum(diff .^ 2, 2);
%     end
%     dist = max(dist, eps);
%     tmp = dist .^ (-1/(s-1));
%     fuzzypartmat = tmp ./ sum(tmp, 2);
% 
%     fuzzypartmat_s = fuzzypartmat .^ s;
%     centers = (fuzzypartmat_s' * data) ./ sum(fuzzypartmat_s, 1)';
% 
%     obj = 0;
%     for k = 1:num_clusters
%         diff = data - centers(k, :);
%         obj = obj + sum(fuzzypartmat_s(:, k) .* sum(diff .^ 2, 2));
%     end
%     objfcn(iter) = obj;
% 
%     if iter > 1 && norm(centers - centers_prev, 'fro') < tol
%         objfcn = objfcn(1:iter);
%         break;
%     end
% end
% end

function [centers, fuzzypartmat, objfcn, W] = fcm_modified(data, num_clusters, options, D)
    if nargin < 2, num_clusters = 2; end
    if nargin < 3, options = [2, 100, 1e-5, 1000]; end % lambda_e = 1000

    m = options(1);
    max_iter = options(2);
    tol = options(3);
    lambda_e = options(4);

    [M, N] = size(data); % e.g., 20 x 6
    data = double(data);

    if any(isnan(data(:))) || any(isinf(data(:)))
        error('Input data contains NaN or Inf');
    end
    if M < num_clusters
        error('Number of data points (%d) is less than number of clusters (%d)', M, num_clusters);
    end
    disp('Data size:'); disp([M, N]);

    % rng('default');
    centers = rand(num_clusters, N) .* range(data) + min(data); % K x 6
    W = ones(1, D) / D; % Initial W is D-dimensional, uniformly distributed
    W_full = repmat(W, 1, round(N/D)); % Expand to N dimensions
    objfcn = zeros(max_iter, 1);

    for iter = 1:max_iter
        centers_prev = centers;
        W_prev = W;

        % Update membership values
        dist = zeros(M, num_clusters);
        for k = 1:num_clusters
            diff = data - centers(k, :);
            dist(:, k) = sum(W_full .* (diff .^ 2), 2);
        end
        dist = max(dist, eps);
        tmp = dist .^ (-1/(m-1));
        fuzzypartmat = (tmp ./ sum(tmp, 2))'; % K x M

        % Update cluster centers
        fuzzypartmat_m = fuzzypartmat .^ m;
        denom = sum(fuzzypartmat_m, 2);
        centers = (fuzzypartmat_m * data) ./ denom; % K x 6

        % Calculate a_d (N-dimensional) and map to D dimensions
        T = M;
        a = zeros(1, N); % N-dimensional
        for d = 1:N
            diff = data(:, d) - centers(:, d)';
            a(d) = sum(sum(fuzzypartmat_m .* (diff .^ 2)'));
        end
        % Map to D dimensions
        a_temp = reshape(a, D, N/D);
        a_3d = sum(a_temp, 2)';
        %a_3d = [a(1) + a(4), a(2) + a(5), a(3) + a(6)]; % x, y, z
        % disp(['Iteration ', num2str(iter), ' - a_d (x, y, z):']);
        % disp(a_3d);

        % Update W (D-dimensional)
        if lambda_e == 0
            W = ones(1, D) / D;
        else
            %a_shifted = a_3d - max(a_3d);
            a_shifted = a_3d;
            exp_terms = exp(-D * a_shifted / (T * lambda_e));
            if all(exp_terms == 0) || any(isnan(exp_terms))
                disp('W calculation failed at iteration:'); disp(iter);
                disp('a_3d:'); disp(a_3d);
                W = W_prev;
            else
                W = exp_terms / sum(exp_terms);
            end
        end
        W_full = repmat(W, 1, round(N/D)); % Expand to N dimensions
        % disp(['Iteration ', num2str(iter), ' - W (x, y, z):']);
        % disp(W);

        % Compute the objective function
        obj = 0;
        for k = 1:num_clusters
            diff = data - centers(k, :);
            obj = obj + sum(fuzzypartmat_m(k, :) .* sum(W_full .* (diff .^ 2), 2)');
        end
        obj = obj / T + (lambda_e / D) * sum(W .* log(max(W, eps)));
        objfcn(iter) = obj;

        if iter > 1 && norm(centers - centers_prev, 'fro') < tol && norm(W - W_prev) < tol
            objfcn = objfcn(1:iter);
            break;
        end
    end
    % disp('Cluster centers:');
    % disp(centers);
end

% function [centers, fuzzypartmat, objfcn, W] = fcm_modified(data, num_clusters, options, D, alpha)
%     if nargin < 2, num_clusters = 2; end
%     if nargin < 3, options = [2, 100, 1e-5, 1000]; end % lambda_e = 1000
%     if nargin < 5, alpha = ones(1, D); end  % Default prior factors are all 1
% 
%     m = options(1);
%     max_iter = options(2);
%     tol = options(3);
%     lambda_e = options(4);
% 
%     [M, N] = size(data); % e.g., 20 x 6
%     data = double(data);
% 
%     if any(isnan(data(:))) || any(isinf(data(:)))
%         error('Input data contains NaN or Inf');
%     end
%     if M < num_clusters
%         error('Number of data points (%d) is less than number of clusters (%d)', M, num_clusters);
%     end
%     disp('Data size:'); disp([M, N]);
% 
%     rng('default');
%     centers = rand(num_clusters, N) .* range(data) + min(data); % K x N
%     W = ones(1, D) / D; % Initial W is D-dimensional, uniformly distributed
%     W_full = repmat(W, 1, round(N / D)); % Expand to N dimensions
%     objfcn = zeros(max_iter, 1);
% 
%     for iter = 1:max_iter
%         centers_prev = centers;
%         W_prev = W;
% 
%         % Update membership values: calculate weighted distances
%         dist = zeros(M, num_clusters);
%         for k = 1:num_clusters
%             diff = data - centers(k, :);
%             dist(:, k) = sum(W_full .* (diff .^ 2), 2);
%         end
%         dist = max(dist, eps);
%         tmp = dist .^ (-1/(m - 1));
%         fuzzypartmat = (tmp ./ sum(tmp, 2))';  % K x M
% 
%         % Update cluster centers
%         fuzzypartmat_m = fuzzypartmat .^ m;
%         denom = sum(fuzzypartmat_m, 2);
%         centers = (fuzzypartmat_m * data) ./ denom; % K x N
% 
%         % Calculate a_d (N-dimensional), then map to D dimensions (e.g., for segmented data, N=6, D=3)
%         T = M;
%         a = zeros(1, N);  % N-dimensional
%         for d = 1:N
%             diff = data(:, d) - centers(:, d)'; % centers(:, d) is 1 x K
%             a(d) = sum(sum(fuzzypartmat_m .* (diff .^ 2)'));
%         end
%         % Map a to D dimensions (assuming each D corresponds to N/D features)
%         a_temp = reshape(a, D, round(N / D));
%         a_3d = sum(a_temp, 2)';  % Results in 1 x D
% 
%         % Introduce prior factor alpha: reduce a_d for less important variables
%         a_3d = a_3d .* alpha; 
% 
%         disp(['Iteration ', num2str(iter), ' - a_d (x, y, z):']);
%         disp(a_3d);
% 
%         % Update W (D-dimensional)
%         if lambda_e == 0
%             W = ones(1, D) / D;
%         else
%             % Optionally, shift a_3d
%             a_shifted = a_3d;  
%             exp_terms = exp(-D * a_shifted / (T * lambda_e));
%             if all(exp_terms == 0) || any(isnan(exp_terms))
%                 disp('W calculation failed at iteration:'); disp(iter);
%                 disp('a_3d:'); disp(a_3d);
%                 W = W_prev;
%             else
%                 W = exp_terms / sum(exp_terms);
%             end
%         end
%         W_full = repmat(W, 1, round(N / D)); % Expand to N dimensions
%         disp(['Iteration ', num2str(iter), ' - W (x, y, z):']);
%         disp(W);
% 
%         % Compute the objective function
%         obj = 0;
%         for k = 1:num_clusters
%             diff = data - centers(k, :);
%             obj = obj + sum(fuzzypartmat_m(k, :) .* sum(W_full .* (diff .^ 2), 2)');
%         end
%         obj = obj / T + (lambda_e / D) * sum(W .* log(max(W, eps)));
%         objfcn(iter) = obj;
% 
%         if iter > 1 && norm(centers - centers_prev, 'fro') < tol && norm(W - W_prev) < tol
%             objfcn = objfcn(1:iter);
%             break;
%         end
%     end
%     disp('Cluster centers:');
%     disp(centers);
% end
