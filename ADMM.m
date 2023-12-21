function [solution, adjacency] = ADMM(lambda1, lambda2, H, p, K, n, m, ...
    w, eta, rho, tol, maxiter, incr, decr)
% the ADMM algorithm to solve the separated problem on a subset of
% variables
% H: input sub-matrix of H
% incr, decr: modify rho to accelerate convergence

%% initialize
N = sum(n); 
M = sum(m); 
Z = zeros(K, M, M); 
for k = 1:K
    Z(k,:,:) = eye(M); 
end
U = zeros(K, M, M); 
newTheta = Z; 

%% ADMM
temp = cumsum(m); 
for iter = 1:maxiter
    % update Theta
    for k = 1:K
        A = rho*(Z(k,:,:)-U(k,:,:))-n(k)/N*H(k,:,:); 
        [V, D] = eig(squeeze(A)); 
        newTheta(k,:,:) = V*(D + sqrt(D^2+4*rho*n(k)/N*eye(M)))*V' ./ (2*rho);
    end
    
    % update Z
    newZ = Z;
    for i = 1:p
        i_lower = temp(i)-m(i)+1; 
        i_upper = temp(i); 
        newZ(:,i_lower:i_upper,i_lower:i_upper) = ...
            newTheta(:,i_lower:i_upper,i_lower:i_upper) + ...
            U(:,i_lower:i_upper,i_lower:i_upper); 
    end
    
    for i = 1:(p-1)
        i_lower = temp(i)-m(i)+1; 
        i_upper = temp(i); 
        for j = (i+1):p
            j_lower = temp(j)-m(j)+1; 
            j_upper = temp(j); 
            bigsoft = double(0);
            for k = 1:K
                A = newTheta(k,i_lower:i_upper,j_lower:j_upper) + ...
                    U(k,i_lower:i_upper,j_lower:j_upper); 
                bigsoft = bigsoft + soft(norm(A(:), 2), lambda1*eta(k,i,j)/rho)^2;
            end
            bigsoft = sqrt(bigsoft); 
            if bigsoft <= lambda2*w(i,j)/rho
                newZ(:,i_lower:i_upper,j_lower:j_upper) = 0;
                newZ(:,j_lower:j_upper,i_lower:i_upper) = 0;
            else
                for k = 1:K
                    A = newTheta(k,i_lower:i_upper,j_lower:j_upper) + ...
                        U(k,i_lower:i_upper,j_lower:j_upper); 
                    A = squeeze(A); 
                    a = soft(norm(A(:), 2), lambda1*eta(k,i,j)/rho); 
                    if a == 0
                        newZ(k,i_lower:i_upper,j_lower:j_upper) = 0;
                        newZ(k,j_lower:j_upper,i_lower:i_upper) = 0;
                    else
                        b = 1 - lambda2*w(i,j)/rho/bigsoft;
                        c = 1 / norm(A(:), 2);
                        newZ(k,i_lower:i_upper,j_lower:j_upper) = a*b*c*A;
                        newZ(k,j_lower:j_upper,i_lower:i_upper) = a*b*c*A';
                    end
                end
            end
        end
    end
    
    % dual residual
    g = rho*(Z(:)-newZ(:)); 
    % primal residual
    r = newTheta(:) -newZ(:); 
    
    % update U
    U = U + newTheta - newZ; 
    
    % check convergence
    epsilon.dual = tol*rho*norm(U(:), 2);
    epsilon.pri = tol*max([norm(newTheta(:), 2), norm(newZ(:), 2)]); 
    if norm(g, 2) <= epsilon.dual && norm(r, 2) <= epsilon.pri
        break
    end
    
    Z = newZ; 
    
    % update rho
    if norm(r, 2)/epsilon.pri > 30*norm(g, 2)/epsilon.dual
        rho = rho*incr;
    end
    if norm(g, 2)/epsilon.dual > 30*norm(r, 2)/epsilon.pri
        rho = rho/decr;
    end
end

%% get adjacency matrix
solution = newZ; 
adjacency = zeros(K, p, p); 
for k = 1:K
    for i = 1:(p-1)
        i_lower = temp(i)-m(i)+1; 
        i_upper = temp(i); 
        for j = (i+1):p
            j_lower = temp(j)-m(j)+1; 
            j_upper = temp(j); 
            if any(abs(solution(k,i_lower:i_upper,j_lower:j_upper)) > 1e-15)
                adjacency(k, i, j) = 1;
                adjacency(k, j, i) = 1;
            end
        end
    end
end


