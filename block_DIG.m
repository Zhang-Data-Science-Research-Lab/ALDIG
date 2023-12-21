function [V, Theta, mu, sigma, newl, adjacency, time, conn] = block_DIG(phi, lambda1, lambda2, p, ...
    l, K, m, n, w, eta, rho, tol, maxiter, incr, decr)
% block version approximate DIG
% phi: 1*K cell array of sufficient statistics. unstandardized
% l: the diagonal used to modify the empirical covariance matrix, a sum(m)-dimensional vector
% m: p-dimensional vector
% n: K-dimensional vector
% w: p*p weight matrix
% eta: K*p*p weight array
% rho, tol, maxiter, incr, decr: ADMM convergence control

N = sum(n);
adjacency = zeros(K, p, p); % K*p*p K adjacency matrices output
V = zeros(K, sum(m), sum(m)); 
Theta = zeros(K, sum(m)+1, sum(m)+1); 
c1 = clock;

%% standardize phi and D
mu = zeros(1, sum(m)); 
sigma = ones(1, sum(m)); 
newl = l; 
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
    newl(i_lower:i_upper) = l(i_lower:i_upper) ./ max(sigma(i_lower:i_upper).^2); 
end

%% construct H
D = diag(newl); 
H = zeros(K, sum(m), sum(m)); 
for k = 1:K
    H_k = phi{k}' * phi{k} ./ n(k); 
    mu0 = phi{k}' * ones(n(k), 1) ./ n(k); 
    H(k,:,:) = H_k - mu0*mu0' + D; 
end

%% verify the Theorem 1 condition
T = eye(p); 
temp = cumsum(m); 
for i = 1:(p-1)
    i_lower = temp(i)-m(i)+1; 
    i_upper = temp(i); 
    for j = (i+1):p
        j_lower = temp(j)-m(j)+1; 
        j_upper = temp(j); 
        s = 0; 
        for k = 1:K
            H_kij = H(k,i_lower:i_upper,j_lower:j_upper); 
            a = n(k)/N*norm(H_kij(:), 2); 
            if a > lambda1*eta(k,i,j)
                s = s + (a - lambda1*eta(k,i,j))^2; 
            end
        end
        if s > lambda2^2 * w(i,j)^2
            T(i,j) = 1; 
            T(j,i) = 1; 
        end
    end
end

%% Tarjan's graph component searching algorithm
G = graph(T); 
conn = conncomp(G); % p-variate graph component indices

%% solve the problem separately using ADMM
for i = 1:max(conn)
    m_g = m(conn == i);
    w_g = w(conn == i, conn == i); 
    eta_g = eta(:, conn == i, conn == i); 
    p_g = sum(conn == i); 
    index = []; 
    for j = find(conn == i)
        j_lower = temp(j)-m(j)+1; 
        j_upper = temp(j); 
        index = [index, j_lower:j_upper]; 
    end
    H_g = H(:,index,index);
    if sum(conn == i) > 1
        [V_g, adjacency_g] = ADMM(lambda1, lambda2, H_g, p_g, K, n, m_g, ...
            w_g, eta_g, rho, tol, maxiter, incr, decr); 
        V(:,index,index) = V_g; 
        adjacency(:,conn == i, conn == i) = adjacency_g; 
    else
        for k = 1:K
            V(k,index,index) = inv(squeeze(H_g(k,:,:))); 
            adjacency(k,conn == i, conn == i) = 0; 
        end
    end
end

c2 = clock; 
time = etime(c2, c1); 

%% convert V to Theta
for k = 1:K
    mu0 = phi{k}' * ones(n(k), 1) ./ n(k); 
    nu = 1 + mu0'*squeeze(V(k,:,:))*mu0; 
    Theta(k,:,:) = [nu, -mu0'*squeeze(V(k,:,:)); -squeeze(V(k,:,:))*mu0, squeeze(V(k,:,:))]; 
end






