clc; clear all;
fprintf('\n================================================================\n');
fprintf('              DEFLATION ALGORITHM            \n');
fprintf('       Finding the Second Dominant Eigenvalue                    \n');
fprintf('================================================================\n\n');
A = [2 0;0 8];
fprintf('>> Input Matrix A:\n\n');
disp(A);
fprintf('\n');
max_iter = 150;    
tol = 1e-12;   

fprintf('>> Algorithm Parameters:\n');
fprintf('   * Maximum iterations: %d\n', max_iter);
fprintf('   * Convergence tolerance: %.0e\n\n', tol);

fprintf('>> Running Power Method on Matrix A to find dominant eigen-pair...\n');
[lambda, v] = powerMethodWithRayleigh(A, max_iter, tol);

if isempty(lambda)
    fprintf('\n   [ERROR] Could not find the dominant eigenvalue. Halting.\n');
    return;
end

fprintf('\n>> Dominant Eigen-pair (lambda_1, v_1) Found:\n');
fprintf('   * lambda_1 = %.10f\n', lambda);
fprintf('   * v_1      = [');
fprintf('%.7f', v(1));
for i = 2:length(v)
    fprintf('; %.7f', v(i));
end
fprintf(']\n\n');
fprintf('================================================================\n');
fprintf('       APPLYING WIELANDT''S DEFLATION                             \n');
fprintf('================================================================\n\n');

x_initial_for_B = ones(size(A, 1) - 1, 1);
[mu, u] = wielandtDeflation(A, lambda, v, x_initial_for_B, max_iter, tol);

if ~isempty(mu)
    fprintf('\n================================================================\n');
    fprintf('                      FINAL RESULTS                            \n');
    fprintf('================================================================\n\n');
    
    fprintf('>> Second Dominant Eigenvalue (lambda_2):\n');
    fprintf('   mu = %.10f\n\n', mu);

    fprintf('>> Corresponding Eigenvector (u_2):\n');
    fprintf('   u = [');
    fprintf('%.7f', u(1));
    for i = 2:length(u)
        fprintf('; %.7f', u(i));
    end
    fprintf(']\n\n');
    
    residual = A * u - mu * u;   % Verification: Check A*u ≈ μ*u
    residual_norm = norm(residual);

    fprintf('>> VERIFICATION (A*u = mu*u):\n');
    fprintf('   * ||A*u - mu*u|| = %.6e\n\n', residual_norm);

    % --- Verification using MATLAB's built-in eig() function ---
    all_eigenvalues = sort(eig(A), 'descend', 'ComparisonMethod', 'abs');
    fprintf('>> COMPARISON WITH MATLAB eig():\n');
    fprintf('   - Wielandt''s Result:    %.10f\n', mu);
    fprintf('   - MATLAB''s 2nd Eig:   %.10f\n', all_eigenvalues(2));
    fprintf('   - Absolute Difference:  %.6e\n\n', abs(mu - all_eigenvalues(2)));
end

function [mu, u] = wielandtDeflation(A, lambda, v, x, max_iter, tolerance)
    n = size(A, 1);
    [~, i] = max(abs(v));
    fprintf('>> Wielandt Step 1: Found max element of v at index i = %d.\n', i);
    A_rows = 1:n; A_rows(i) = [];
    A_cols = 1:n; A_cols(i) = [];    
    A_sub = A(A_rows, A_cols);
    v_sub = v(A_rows);
    A_row_i_sub = A(i, A_cols);
    B = A_sub - (v_sub / v(i)) * A_row_i_sub;    
    fprintf('>> Wielandt Step 2: Deflated matrix B created.\n');
    fprintf('>> Wielandt Step 3: Running Power Method on B...\n');
    mu = powerMethodWithRayleigh(B, max_iter, tolerance, true);
    [mu, w_prime] = powerMethodWithRayleigh(B, max_iter, tolerance, false);
    if isempty(mu)
        fprintf('\n   [ERROR] Wielandt''s Deflation failed: Power Method on B did not converge.\n');
        u = [];
        return;
    end
    fprintf('\n   [SUCCESS] Power Method on B converged. Eigenvalue mu = %.6f\n', mu);
    w = zeros(n, 1);
    w(A_rows) = w_prime;
    sum_term = A(i, :) * w;
    u = (mu - lambda) * w + sum_term * (v / v(i));    
    fprintf('>> Wielandt Step 4: Final eigenvector `u` constructed.\n');
end

function [eigenvalue, eigenvector] = powerMethodWithRayleigh(A, mx, t, suppress_output)
    if nargin < 4, suppress_output = false; end % Default to show output
    n = size(A, 1);
    x = ones(n, 1);
    x = x / norm(x);    
    if ~suppress_output
        fprintf('\n>> POWER METHOD ITERATION PROGRESS:\n\n');
        fprintf('  Iteration  |  ||x_(k+1) - x_k||_2  |   Status\n');
        fprintf('  -----------+----------------------+---------------------\n');
    end    
    for i = 1:mx
        Ax = A * x;
        x_new = Ax / norm(Ax);
        change = norm(x_new - x);
        
        if ~suppress_output && (i <= 5 || mod(i, 10) == 0 || change < t)
            status = '';
            if change < t, status = '-> CONVERGED [OK]'; end
            fprintf('  %9d  |  %.10e  |  %s\n', i, change, status);
        end        
        if change < t
            if ~suppress_output
                fprintf('\n   [SUCCESS] Eigenvector convergence after %d iterations\n', i);
            end
            eigenvector = x_new;
            eigenvalue = rayleighQuotient(A, eigenvector);
            return;
        end
        x = x_new;
    end    
    if ~suppress_output
        fprintf('\n   [WARNING] Maximum iterations (%d) reached\n', mx);
    end
    eigenvector = x;
    eigenvalue = rayleighQuotient(A, eigenvector);
end
function lambda = rayleighQuotient(A, x)
    lambda = (x' * A * x) / (x' * x);
end