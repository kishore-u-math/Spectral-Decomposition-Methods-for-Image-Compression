clc; clear all;

%% ========================================================================
%         LANCZOS ITERATION FOR EIGENVALUE APPROXIMATION
%% ========================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('        LANCZOS ITERATION FOR EIGENVALUE APPROXIMATION         \n');
fprintf('================================================================\n\n');

%% ------------------------------------------------------------------------
%                         TEST MATRIX AND PARAMETERS
%% ------------------------------------------------------------------------
% Lanczos requires the matrix to be symmetric.
rng(1); % for reproducibility
A_rand = rand(20);
A = A_rand + A_rand'; % A simple way to create a symmetric matrix

fprintf('>> Input Matrix A:\n');
fprintf('   * A 20x20 random symmetric matrix has been created.\n\n');

% Number of Lanczos iterations
m = 15;
fprintf('>> Algorithm Parameters:\n');
fprintf('   * Number of Lanczos Iterations (m): %d\n', m);

% Initial vector
b = rand(size(A, 1), 1);
fprintf('   * An initial random vector `b` has been created.\n\n');

%% ------------------------------------------------------------------------
%                    CALL LANCZOS ITERATION FUNCTION
%% ------------------------------------------------------------------------

fprintf('================================================================\n');
fprintf('              RUNNING LANCZOS ITERATION                      \n');
fprintf('================================================================\n');

[Q, T] = lanczosIteration(A, b, m);

%% ------------------------------------------------------------------------
%          FINAL EIGENVALUE & EIGENVECTOR CALCULATION
%% ------------------------------------------------------------------------

fprintf('\n================================================================\n');
fprintf('   FINAL EIGENVALUE & EIGENVECTOR RESULTS (m = %d iterations)  \n', m);
fprintf('================================================================\n\n');

% Use my_qr to get eigenvalues (Ritz values) and eigenvectors of T
qr_iterations = 100;
qr_tolerance = 1e-12;
[ritz_values, S] = my_qr(T, qr_iterations, qr_tolerance);

% Sort Ritz values and corresponding eigenvectors of T
[sorted_ritz_values, sort_idx] = sort(ritz_values, 'descend');
sorted_S = S(:, sort_idx);

fprintf('>> Final Approximated Eigenvalues from T_%d (sorted descending):\n\n', m);
disp(sorted_ritz_values);
fprintf('\n');

% Construct the Ritz vectors (approximations of A's eigenvectors)
% Y = Q * S, where S are the eigenvectors of T
Ritz_vectors = Q * sorted_S;

%% ------------------------------------------------------------------------
%                      VERIFICATION OF RESULTS
%% ------------------------------------------------------------------------

fprintf('================================================================\n');
fprintf('             VERIFICATION OF LANCZOS PROCESS                     \n');
fprintf('================================================================\n\n');

% 1. Verify the Lanczos Relation: T = Q' * A * Q
T_reconstructed = Q' * A * Q;
reconstruction_error = norm(T - T_reconstructed, 'fro');

fprintf('>> Verifying the Lanczos Relation T = Q'' * A * Q:\n');
fprintf('   * The error ||T - Q''AQ||_F = %.4e\n', reconstruction_error);
fprintf('   * Note: This "error" is expected and represents the residual of the truncated process.\n\n');

% 2. Verify the Ritz Vectors (the correct way to check reconstruction)
fprintf('>> Verifying the Ritz Vectors (Approximated Eigenvectors of A):\n');
fprintf('   Checking the residual ||A*y_i - lambda_i*y_i|| for the top 5 pairs...\n\n');
for i = 1:5
    lambda_i = sorted_ritz_values(i);
    y_i = Ritz_vectors(:, i);
    residual_norm = norm(A*y_i - lambda_i*y_i);
    fprintf('   * For Eigenvalue Approx %.6f, Residual Norm = %.4e\n', lambda_i, residual_norm);
end
fprintf('\n');


fprintf('================================================================\n');
fprintf('             VERIFICATION WITH MATLAB''S BUILT-IN eig()          \n');
fprintf('================================================================\n\n');

true_eigenvalues = sort(eig(A), 'descend');
fprintf('>> Top %d True Eigenvalues of Original Matrix A (from eig()):\n\n', m);
disp(true_eigenvalues(1:10));
fprintf('\n');

fprintf('>> Comparison of Top 10 Approximated Eigenvalues:\n');
fprintf('   +----------------------------------------------------------+\n');
fprintf('   |   Lanczos Approx.   |   True Eigenvalue   | Difference   |\n');
fprintf('   +----------------------------------------------------------+\n');
for i = 1:10
    fprintf('   |   %15.8f   |   %15.8f   |  %.4e  |\n', sorted_ritz_values(i), true_eigenvalues(i), abs(sorted_ritz_values(i) - true_eigenvalues(i)));
end
fprintf('   +----------------------------------------------------------+\n\n');

%% ========================================================================
%             USER-DEFINED FUNCTION FOR LANCZOS ITERATION
%% ========================================================================
function [Q, T] = lanczosIteration(A, b, m)
    n = size(A, 1);
    Q = zeros(n, m);
    T = zeros(m, m);
    
    q_prev = zeros(n, 1);
    beta = 0;
    q_curr = b / norm(b);
    
    for j = 1:m
        Q(:, j) = q_curr;
        v = A * q_curr;
        alpha = q_curr' * v;
        T(j, j) = alpha;
        
        v = v - beta * q_prev - alpha * q_curr;
        beta = norm(v);
        
        if j < m
            T(j, j+1) = beta;
            T(j+1, j) = beta;
            if abs(beta) < 1e-12
                fprintf('\n   [INFO] Lucky breakdown at iteration %d.\n', j);
                T = T(1:j, 1:j); Q = Q(:, 1:j);
                break;
            end
            q_prev = q_curr;
            q_curr = v / beta;
        end
        
        fprintf('\n>> Eigenvalue Estimates at Iteration m = %d\n', j);
        T_current = T(1:j, 1:j);
        [estimated_eigs, ~] = my_qr(T_current, 50, 1e-7);
        disp(sort(estimated_eigs, 'descend'));
    end
end

%% ========================================================================
%             USER-DEFINED QR ALGORITHM FOR EIGEN-DECOMPOSITION
%% ========================================================================
function [eigenvalues, eigenvectors] = my_qr(A, N, TOL)
    A_current = A;
    eigenvectors = eye(size(A));
    for k = 1:N
        [Q_iter, R_iter] = householderQR(A_current);
        A_current = R_iter * Q_iter;
        eigenvectors = eigenvectors * Q_iter;
        if sum(abs(tril(A_current, -1)), 'all') < TOL, break; end
    end
    eigenvalues = diag(A_current);
end

%% ========================================================================
%             USER-DEFINED HOUSEHOLDER QR DECOMPOSITION
%% ========================================================================
function [Q, R] = householderQR(A)
    [m, n] = size(A);
    Q_acc = eye(m);
    R = A;
    for j = 1:min(m-1, n)
        x = R(j:m, j);
        v = householderVector(x);
        R_sub = R(j:m, j:n);
        R(j:m, j:n) = R_sub - 2 * v * (v' * R_sub);
        Q_sub = Q_acc(j:m, :);
        Q_acc(j:m, :) = Q_sub - 2 * v * (v' * Q_sub);
    end
    Q = Q_acc';
end

%% ========================================================================
%             HELPER FUNCTION TO CREATE HOUSEHOLDER VECTOR
%% ========================================================================
function v = householderVector(x)
    n = length(x);
    s = sign(x(1));
    if s == 0, s = 1; end
    norm_x = norm(x);
    u1 = x(1) + s * norm_x;
    u = [u1; x(2:n)];
    v = u / norm(u);
end
