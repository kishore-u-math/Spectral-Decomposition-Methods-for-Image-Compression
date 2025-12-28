clc; clear all;
fprintf('\n================================================================\n');
fprintf('        QR ALGORITHM FOR EIGENVALUE & EIGENVECTOR COMPUTATION   \n');
fprintf('================================================================\n\n');
B=rand(6); A= B+B';
fprintf('>> Input Matrix A:\n\n');
disp(A);
max_iter = 1000;
tol = 1e-12;
fprintf('\n>> Algorithm Parameters:\n');
fprintf('   * Maximum iterations: %d\n', max_iter);
fprintf('   * Convergence tolerance: %.0e\n\n', tol);
[eigenvalues, eigenvectors] = my_qr(A, max_iter, tol);
fprintf('\n================================================================\n');
fprintf('              EIGENVALUES AND EIGENVECTORS           \n');
fprintf('================================================================\n\n');
[sorted_eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
sorted_eigenvectors = eigenvectors(:, sort_idx);
for i = 1:length(sorted_eigenvalues)
    fprintf('>> Eigenvalue %d (lambda_%d):\n', i, i);
    fprintf('   Value: %.10f\n\n', sorted_eigenvalues(i));
    fprintf('>> Corresponding Eigenvector %d (v_%d):\n\n', i, i);
    disp(sorted_eigenvectors(:, i));
    fprintf('\n----------------------------------------------------------------\n\n');
end
fprintf('================================================================\n');
fprintf('                       VERIFICATION                            \n');
fprintf('================================================================\n\n');
for i = 1:length(sorted_eigenvalues)
    lambda = sorted_eigenvalues(i);
    v = sorted_eigenvectors(:, i);
    residual_norm = norm(A*v - lambda*v);    
    fprintf('>> Verifying Eigen-pair %d:\n', i);
    fprintf('   * ||A*v_%d - lambda_%d*v_%d|| = %.4e\n', i, i, i, residual_norm);
    if residual_norm < 1e-6
        fprintf('   * Status: Verified [OK]\n\n');
    else
        fprintf('   * Status: High Error [CHECK]\n\n');
    end
end
fprintf('================================================================\n\n');
function [eigenvalues, eigenvectors] = my_qr(A, N, TOL)
    A_current = A;
    eigenvectors = eye(size(A));
    fprintf('\n>> QR ALGORITHM ITERATION PROGRESS:\n\n');
    fprintf('  Iteration  |  Max Off-Diagonal Value\n');
    fprintf('  -----------+--------------------------\n');    
    for k = 1:N
        [Q, R] = householderQR(A_current);
        A_next = R * Q;
        eigenvectors = eigenvectors * Q;
        A_current = A_next;
        off_diag_val = max(abs(tril(A_current, -1)), [], 'all');
        if k <= 10 || mod(k, 10) == 0 || off_diag_val < TOL
            fprintf('  %9d  |  %.10e\n', k, off_diag_val);
        end
        if off_diag_val < TOL
            fprintf('\n   [SUCCESS] Convergence reached after %d iterations.\n', k);
            break;
        end
    end    
    if k == N
        fprintf('\n   [WARNING] Maximum number of iterations (%d) reached.\n', N);
    end
    eigenvalues = diag(A_current);
end
function [Q, R] = householderQR(A)
    [m, n] = size(A);
    Q = eye(m);
    R = A;    
    for j = 1:min(m-1, n)
        x = R(j:m, j);
        v = householderVector(x);
        R_sub = R(j:m, j:n);
        R(j:m, j:n) = R_sub - 2 * v * (v' * R_sub);
        Q_sub = Q(j:m, :);
        Q(j:m, :) = Q_sub - 2 * v * (v' * Q_sub);
    end    
    Q = Q';
end
function v = householderVector(x)
    n = length(x);
    s = sign(x(1));
    if s == 0, s = 1; end
    norm_x = norm(x);
    u1 = x(1) + s * norm_x;
    u = [u1; x(2:n)];
    v = u / norm(u);
end
