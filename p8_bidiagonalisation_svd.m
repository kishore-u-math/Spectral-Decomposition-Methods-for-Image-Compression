clear all; close all; clc;

% Random Test matrix (Rectangular)
B = rand(5, 7);
fprintf('Test matrix B (%dx%d):\n', size(B));
disp(B);

%% Bidiagonalize the matrix
[U, Bidiag, V] = bidiagonalize(B);
fprintf('\nBidiagonal matrix form:\n');
disp(Bidiag);
%% Compute SVD using my_svd on the bidiagonal matrix
[U1, S1, V1] = my_svd(Bidiag);

% Combine transformations
U_final = U * U1;
V_final = V * V1;
S_final = S1;

fprintf('\n========== MY SVD RESULTS ==========\n');
fprintf('U matrix (%dx%d):\n', size(U_final, 1), size(U_final, 2));
disp(U_final);

fprintf('\nS matrix (%dx%d):\n', size(S_final, 1), size(S_final, 2));
disp(S_final);

fprintf('\nV matrix (%dx%d):\n', size(V_final, 1), size(V_final, 2));
disp(V_final);

% Verify reconstruction
A_recon1 = U_final * S_final * V_final'
fprintf('\nMy SVD reconstruction error: %.15e\n\n', norm(B - A_recon1, 'fro'));

% Comparing with MATLAB's built-in SVD
[U2, S2, V2] = svd(B);

fprintf('========== MATLAB BUILT-IN SVD ==========\n');
fprintf('U matrix (%dx%d):\n', size(U2, 1), size(U2, 2));
disp(U2);

fprintf('\nS matrix (%dx%d):\n', size(S2, 1), size(S2, 2));
disp(S2);

fprintf('\nV matrix (%dx%d):\n', size(V2, 1), size(V2, 2));
disp(V2);

A_original = U2 * S2 * V2';
fprintf('\nMATLAB svd() reconstruction error: %.15e\n\n', norm(B - A_original, 'fro'));

% =========================================================================
function [U, S, V] = my_svd(A)
    
    [m, n] = size(A);
    
    if m >= n
        %% Case 1: m >= n (tall or square matrix)
        ATA = A' * A;
        [V_temp, D] = eig(ATA);
        [d, idx] = sort(diag(D), 'descend');
        V = V_temp(:, idx);
        
        sigma = sqrt(max(0, d));
        
        S = zeros(m, n);
        for i = 1:n
            S(i, i) = sigma(i);
        end
        
        U_partial = zeros(m, n);
        for i = 1:n
            if sigma(i) > 1e-10
                U_partial(:, i) = (A * V(:, i)) / sigma(i);
            end
        end
        
        if m > n
            U = complete_basis_gramschmidt(U_partial, m);
        else
            U = U_partial;
        end
        
    else
        %% Case 2: m < n (wide matrix)
        AAT = A * A';
        [U_temp, D] = eig(AAT);
        [d, idx] = sort(diag(D), 'descend');
        U = U_temp(:, idx);
        
        sigma = sqrt(max(0, d));
        
        S = zeros(m, n);
        for i = 1:m
            S(i, i) = sigma(i);
        end
        
        V_partial = zeros(n, m);
        for i = 1:m
            if sigma(i) > 1e-10
                V_partial(:, i) = (A' * U(:, i)) / sigma(i);
            end
        end
        
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
    
    for j = k+1:n
        v = randn(m, 1);
        
        for i = 1:j-1
            v = v - (Q(:, i)' * v) * Q(:, i);
        end
        
        v_norm = norm(v);
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

% =========================================================================
function [U, B, V] = bidiagonalize(A)
    
    [m, n] = size(A);
    
    B = A;
    U = eye(m);
    V = eye(n);
    
    max_iter = min(m, n);
    
    for i = 1:max_iter
        %% LEFT HOUSEHOLDER
        x = B(i:m, i);
        v_left = householder_vector(x);
        H_L = eye(m - i + 1) - 2 * (v_left * v_left');
        B(i:m, i:n) = H_L * B(i:m, i:n);
        
        U_temp = eye(m);
        U_temp(i:m, i:m) = H_L;
        U = U * U_temp;
        
        %% RIGHT HOUSEHOLDER
        if i < n
            y = B(i, i+1:n)';
            v_right = householder_vector(y);
            H_R = eye(n - i) - 2 * (v_right * v_right');
            B(i:m, i+1:n) = B(i:m, i+1:n) * H_R;
            
            V_temp = eye(n);
            V_temp(i+1:n, i+1:n) = H_R;
            V = V * V_temp;
        end
    end
end

% =========================================================================
function v = householder_vector(x)
    
    x = x(:);
    norm_x = norm(x);
    
    if norm_x == 0
        v = zeros(size(x));
        return;
    end
    
    sigma = sign(x(1));
    if sigma == 0
        sigma = 1;
    end
    
    mu = x(1) + sigma * norm_x;
    v = x;
    v(1) = mu;
    v = v / norm(v);
end