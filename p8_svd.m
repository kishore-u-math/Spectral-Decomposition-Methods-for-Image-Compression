clear all; close all; clc;

% Test matrix
A = rand(2,5);
fprintf('Test matrix A (%dx%d):\n', size(A));
disp(A);

%% Compute SVD using my_svd
[U1, S1, V1] = my_svd(A);

fprintf('\n My SVD Results:\n');
fprintf('U matrix:\n'); disp(U1);
fprintf('S matrix:\n'); disp(S1);
fprintf('V matrix:\n'); disp(V1);

% Verify reconstruction
A_recon1 = U1 * S1 * V1';
fprintf(' My SVD reconstruction:\n');
disp(A_recon1);
fprintf('Reconstruction error: %.15e\n\n', norm(A - A_recon1, 'fro'));

% Comparing with MATLAB's built-in SVD
[U2, S2, V2] = svd(A);
fprintf('\n Inbuilt SVD Results:\n');
fprintf('U matrix:\n'); disp(U2);
fprintf('S matrix:\n'); disp(S2);
fprintf('V matrix:\n'); disp(V2);
A_original = U2 * S2 * V2';

fprintf(' MATLAB svd() reconstruction:\n');
disp(A_original);
fprintf('Reconstruction error: %.15e\n\n', norm(A - U1*S1*V1', 'fro'));


% =========================================================================
function [U, S, V] = my_svd(A)
    
    [m, n] = size(A);
    
    if m >= n
        %% Case 1: m >= n (tall or square matrix)
        % Step 1: Compute V from eigendecomposition of A'*A
        ATA = A' * A;
        [V_temp, D] = eig(ATA);
        
        % Sort eigenvalues and eigenvectors in descending order
        [d, idx] = sort(diag(D), 'descend');
        V = V_temp(:, idx);
        
        % Step 2: Compute singular values
        sigma = sqrt(max(0, d));  % Handle numerical errors
        
        % Step 3: Create S matrix (m x n)
        S = zeros(m, n);
        for i = 1:n
            S(i, i) = sigma(i);
        end
        
        % Step 4: Compute first n columns of U from A*V = U*Sigma
        U_partial = zeros(m, n);
        for i = 1:n
            if sigma(i) > 1e-10
                U_partial(:, i) = (A * V(:, i)) / sigma(i);
            end
        end
        
        % Step 5: Complete U to m x m using Gram-Schmidt
        if m > n
            U = complete_basis_gramschmidt(U_partial, m);
        else
            U = U_partial;
        end
        
    else
        %% Case 2: m < n (wide matrix)
        % Step 1: Compute U from eigendecomposition of A*A'
        AAT = A * A';
        [U_temp, D] = eig(AAT);
        
        % Sort eigenvalues and eigenvectors in descending order
        [d, idx] = sort(diag(D), 'descend');
        U = U_temp(:, idx);
        
        % Step 2: Compute singular values
        sigma = sqrt(max(0, d));
        
        % Step 3: Create S matrix (m x n)
        S = zeros(m, n);
        for i = 1:m
            S(i, i) = sigma(i);
        end
        
        % Step 4: Compute first m columns of V from A'*U = V*Sigma
        V_partial = zeros(n, m);
        for i = 1:m
            if sigma(i) > 1e-10
                V_partial(:, i) = (A' * U(:, i)) / sigma(i);
            end
        end
        
        % Step 5: Complete V to n x n using Gram-Schmidt
        if n > m
            V = complete_basis_gramschmidt(V_partial, n);
        else
            V = V_partial;
        end
    end
end

% =========================================================================
function Q = complete_basis_gramschmidt(Q_partial, n)
    
    [m, k] = size(Q_partial);
    Q = [Q_partial, zeros(m, n - k)];
    
    % Generate random vectors and orthogonalize them
    for j = k+1:n
        % Start with a random vector
        v = randn(m, 1);
        
        % Orthogonalize against all previous columns using Gram-Schmidt
        for i = 1:j-1
            v = v - (Q(:, i)' * v) * Q(:, i);
        end
        
        % Normalize
        v_norm = norm(v);
        
        % If norm is too small, try another random vector
        attempts = 0;
        while v_norm < 1e-10 && attempts < 10
            v = randn(m, 1);
            for i = 1:j-1
                v = v - (Q(:, i)' * v) * Q(:, i);
            end
            v_norm = norm(v);
            attempts = attempts + 1;
        end
        
        Q(:, j) = v / v_norm;
    end
end
