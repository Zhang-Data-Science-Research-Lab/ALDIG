function [alpha_opt, lambda_opt, lambda_1se, cv_err, rtime] = cv(nfold, ...
    phi, alpha, lambda, p, l, K, m, n, w, eta, rho, tol, maxiter, ...
    incr, decr, seed)
% cross validation for the ALDIG method
% phi is a 1*K cell array
% n is a K-dimensional vector

%% permute full data to get random split of training set and validation set
rng(seed); 
for k = 1:K
    ind = randperm(n(k)); 
    phi{k} = phi{k}(ind,:); 
end
t1 = clock; 

%% cv
cv_err = zeros(length(alpha), length(lambda), nfold); 
for fold = 1:nfold
    v_n = zeros(1, K); 
    t_n = zeros(1, K); 
    t_phi = cell(1, K); 
    % training set
    for k = 1:K
        a = fold:nfold:n(k); 
        b = setdiff(1:n(k), 1); 
        t_phi{k} = phi{k}(b,:); 
        v_n(k) = length(a); 
        t_n(k) = length(b); 
    end
    
    for i = 1:length(alpha)
        for j = 1:length(lambda)
            lambda1 = lambda(j)*(1-alpha(i)); 
            lambda2 = lambda(j)*alpha(i); 
            [~, Theta, mu, sigma, t_l] = block_DIG(t_phi, lambda1, lambda2, p, ...
                l, K, m, t_n, w, eta, rho, tol, maxiter, incr, decr); 
            % validation set
            v_phi = cell(1, K); 
            for k = 1:K
                a = fold:nfold:n(k); 
                v_phi{k} = phi{k}(a,:); 
                for r = 1:sum(m)
                    v_phi{k}(:,r) = (v_phi{k}(:,r) - mu(r))./sigma(r); 
                end
            end
            % construct Sigma from validation data
            D = diag(t_l); 
            for k = 1:K
                H = v_phi{k}'*v_phi{k} ./ v_n(k) + D; 
                mu0 = v_phi{k}'*ones(v_n(k),1) ./ v_n(k); 
                Sigma = [1, mu0'; mu0, H];    % sum(m)+1 by sum(m)+1 matrix
                A = trace(reshape(Theta(k,:,:), sum(m)+1, sum(m)+1) * Sigma) - ...
                    log(det(reshape(Theta(k,:,:), sum(m)+1, sum(m)+1))); 
                cv_err(i,j,fold) = cv_err(i,j,fold) + A * v_n(k) / sum(v_n); 
            end       
        end
    end
end

%% tune the optimal alpha
cv_score = mean(cv_err, 3); 
[~, index] = min(cv_score(:)); 
[a, b] = ind2sub(size(cv_score), index); 
alpha_opt = alpha(a); 
lambda_opt = lambda(b); 

%% 1-se rule
cv_score = cv_score(a,:); 
cv_se = std(cv_err, [], 3) / sqrt(nfold); 
lambda_1se = max(lambda(cv_score <= min(cv_score) + cv_se(a,b))); 
t2 = clock; 
rtime = etime(t2, t1); 



