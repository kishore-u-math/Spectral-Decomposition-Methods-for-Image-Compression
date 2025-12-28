clc; clear all;

%% ========================================================================
%         ARNOLDI ITERATION FOR EIGENVALUE APPROXIMATION
%% ========================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('        ARNOLDI ITERATION FOR EIGENVALUE APPROXIMATION         \n');
fprintf('================================================================\n\n');

%% ------------------------------------------------------------------------
%                         TEST MATRIX AND PARAMETERS
%% ------------------------------------------------------------------------

% A larger matrix to better see the approximation.
%rng(1); % for reproducibility
A = rand(20);
A = A + A'; % Make it symmetric to guarantee real eigenvalues

fprintf('>> Input Matrix A:\n');
fprintf('   * A 20x20 random symmetric matrix has been created.\n\n');

% Number of Arnoldi iterations (size of the Hessenberg matrix)
m = 15;
fprintf('>> Algorithm Parameters:\n');
fprintf('   * Number of Arnoldi Iterations (m): %d\n', m);

% Initial vector
b = rand(size(A, 1), 1);
fprintf('   * An initial random vector `b` has been created.\n\n');

%% ------------------------------------------------------------------------
%                    CALL ARNOLDI ITERATION FUNCTION
%% ------------------------------------------------------------------------

fprintf('================================================================\n');
fprintf('              RUNNING ARNOLDI ITERATION                      \n');
fprintf('================================================================\n');

[Q, H] = arnoldiIteration(A, b, m);

%% ------------------------------------------------------------------------
%              FINAL EIGENVALUE CALCULATION AND DISPLAY
%% ------------------------------------------------------------------------

fprintf('\n================================================================\n');
fprintf('        FINAL EIGENVALUE RESULTS (after m = %d iterations)       \n', m);
fprintf('================================================================\n\n');

% The final Hessenberg matrix from Arnoldi
final_H = H(1:m, 1:m);

% Use the custom QR algorithm to get final eigenvalues and eigenvectors
qr_iterations = 100;
qr_tolerance = 1e-18;
[final_eigenvalues, final_eigenvectors] = my_qr(final_H, qr_iterations, qr_tolerance);
fprintf('>> Final Approximated Eigenvalues from H_%d :\n\n', m);
disp(final_eigenvalues);
fprintf('\n');

%% ------------------------------------------------------------------------
%                      VERIFICATION OF RESULTS
%% ------------------------------------------------------------------------

fprintf('================================================================\n');
fprintf('             VERIFICATION WITH MATLAB''S BUILT-IN eig()          \n');
fprintf('================================================================\n\n');

% Get true eigenvalues from the original matrix A
true_eigenvalues = sort(eig(A), 'descend');
fprintf('>> Top %d True Eigenvalues of Original Matrix A (from eig()):\n\n', m);
disp(true_eigenvalues(1:m));
fprintf('\n');

fprintf('>> Comparison of Top 5 Eigenvalues:\n');
fprintf('   +----------------------------------------------------+\n');
fprintf('   |   Arnoldi Approx.   |   True Eigenvalue   | Difference |\n');
fprintf('   +----------------------------------------------------+\n');
for i = 1:5
    fprintf('   |   %15.8f   |   %15.8f   |  %.4e  |\n', final_eigenvalues(i), true_eigenvalues(i), abs(final_eigenvalues(i) - true_eigenvalues(i)));
end
fprintf('   +----------------------------------------------------+\n\n');

fprintf('================================================================\n\n');

%% ========================================================================
%             USER-DEFINED FUNCTION FOR ARNOLDI ITERATION
%% ========================================================================

function [Q, H] = arnoldiIteration(A, b, m)
    % ARNOLDIITERATION Performs 'm' steps of the Arnoldi iteration.
    n = size(A, 1);
    Q = zeros(n, m + 1);
    H = zeros(m + 1, m);
    
    Q(:, 1) = b / norm(b); % Normalize initial vector
    
    for j = 1:m
        % Matrix-Vector Product
        w = A * Q(:, j);
        
        % Orthogonalization (Gram-Schmidt)
        for i = 1:j
            H(i, j) = Q(:, i)' * w;
            w = w - H(i, j) * Q(:, i);
        end
        
        if j < m
            H(j + 1, j) = norm(w);
            if abs(H(j + 1, j)) < 1e-12
                fprintf('\n   [INFO] Lucky breakdown at iteration %d. Subspace is invariant.\n', j);
                m = j; H = H(1:m, 1:m); Q = Q(:, 1:m);
                break;
            end
            Q(:, j + 1) = w / H(j + 1, j);
        end
        
        % --- Display intermediate estimates ---
        fprintf('\n>> Eigenvalue Estimates at Iteration m = %d\n', j);
        H_current = H(1:j, 1:j);
        [estimated_eigs, ~] = my_qr(H_current, 50, 1e-6); % Discard eigenvectors
        disp(sort(estimated_eigs, 'descend'));
    end
    
    if size(H, 1) > m
        H = H(1:m, 1:m);
    end
end

%% ========================================================================
%             USER-DEFINED QR ALGORITHM FOR EIGEN-DECOMPOSITION
%% ========================================================================

function [eigenvalues, eigenvectors] = my_qr(A, N, TOL)
    % my_qr Finds eigenvalues and eigenvectors of a matrix A.
    A_current = A;
    eigenvectors = eye(size(A)); % Initialize eigenvector matrix

    for k = 1:N
        [Q_iter, R_iter] = householderQR(A_current);
        A_current = R_iter * Q_iter;
        eigenvectors = eigenvectors * Q_iter; % Accumulate transformations
        
        if sum(abs(tril(A_current, -1)), 'all') < TOL
            break; % Converged
        end
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
