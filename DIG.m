function [V, Theta, adjacency, time] = DIG(phi, lambda1, lambda2, p, ...
    l, K, m, n, w, eta, rho, tol, maxiter, incr, decr)
% original approximate DIG
% phi: 1*K cell array of sufficient statistics. unstandardized
% l: the diagonal used to modify the empirical covariance matrix, 
% a sum(m)-dimensional vector
% n: K-dimensional vector
% w: p*p weight matrix
% eta: K*p*p weight array

c1 = clock; 
%% standardize phi and D
mu = zeros(1, sum(m)); 
sigma = ones(1, sum(m)); 
for i = 1:p
    [i_lower, i_upper] = getindex(m, i); 
    for j = i_lower:i_upper
        suff = []; 
        for k = 1:K
            suff = [suff ; phi{k}(:,j)]; 
        end
        mu(j) = mean(suff(:)); 
        sigma(j) = std(suff(:)); 
        for k = 1:K
            phi{k}(:,j) = (phi{k}(:,j) - mu(j)) ./ sigma(j); 
        end
    end
    l(i_lower:i_upper) = l(i_lower:i_upper) ./ max(sigma(i_lower:i_upper).^2); 
end

%% construct H
D = diag(l); 
H = zeros(K, sum(m), sum(m)); 
for k = 1:K
    H_k = phi{k}' * phi{k} ./ n(k); 
    mu0 = phi{k}' * ones(n(k), 1) ./ n(k); 
    H(k,:,:) = H_k - mu0*mu0' + D; 
end

%% solve by ADMM
[V, adjacency] = ADMM(lambda1, lambda2, H, p, K, n, m, ...
            w, eta, rho, tol, maxiter, incr, decr); 

c2 = clock; 
time = etime(c2, c1); 

%% convert V to Theta
Theta = zeros(K, sum(m)+1, sum(m)+1); 
for k = 1:K
    mu0 = phi{k}' * ones(n(k), 1) ./ n(k); 
    nu = 1 + mu0'*squeeze(V(k,:,:))*mu0; 
    Theta(k,:,:) = [nu, -mu0'*squeeze(V(k,:,:)); -squeeze(V(k,:,:))*mu0, squeeze(V(k,:,:))]; 
end

