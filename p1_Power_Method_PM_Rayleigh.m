clc; clear all;
fprintf('\n');
fprintf('================================================================\n');
fprintf('     POWER METHOD WITH RAYLEIGH QUOTIENT ALGORITHM            \n');
fprintf('              Finding Dominant Eigenvalues                    \n');
fprintf('================================================================\n\n');

A = rand(3);       % Create 3x3 random matrix 
fprintf('>> Matrix A:\n\n');
disp(A);
fprintf('\n');
max_iter = 150;      % Maximum number of iterations
tol = 1e-16;         % Convergence tolerance (strict for numerical precision)

fprintf('>> Algorithm Parameters:\n');
fprintf('   * Maximum iterations: %d\n', max_iter);
fprintf('   * Convergence tolerance: %.0e\n\n', tol);
[dominant_eigenvalue, dominant_eigenvector] = ...
    powerMethodWithRayleigh(A, max_iter, tol);
fprintf('\n================================================================\n');
fprintf('                      FINAL RESULTS                            \n');
fprintf('================================================================\n\n');
fprintf('>> Dominant Eigenvector:\n');
fprintf('   v = [');
fprintf('%.10f', dominant_eigenvector(1));
for i = 2:length(dominant_eigenvector)
    fprintf('; %.10f', dominant_eigenvector(i));
end
fprintf(']\n\n');
fprintf('>> Dominant Eigenvalue (via Rayleigh Quotient):\n');
fprintf('   lambda = %.15f\n\n', dominant_eigenvalue);
% Verification: Check A*v ≈ λ*v
residual = A * dominant_eigenvector - dominant_eigenvalue * dominant_eigenvector;
residual_norm = norm(residual);
fprintf('>> VERIFICATION (A*v = lambda*v):\n');
fprintf('   * A*v - lambda*v = [');
fprintf('%.6e', residual(1));
for i = 2:length(residual)
    fprintf('; %.6e', residual(i));
end
fprintf(']\n');
fprintf('   * ||A*v - lambda*v|| = %.6e\n\n', residual_norm);

% Compare with MATLAB's built-in eigenvalues
[V_matlab, D_matlab] = eig(A);
[lambda_matlab, idx] = max(diag(D_matlab));
v_matlab = V_matlab(:, idx);

% --- Simplified Comparison Output ---
fprintf('>> COMPARISON WITH MATLAB eig():\n');
fprintf('   - Power Method Result:  %.10f\n', dominant_eigenvalue);
fprintf('   - MATLAB eig() Result:    %.10f\n', lambda_matlab);
fprintf('   - Absolute Difference:  %.6e\n\n', abs(dominant_eigenvalue - lambda_matlab));

function [eigenvalue, eigenvector] = powerMethodWithRayleigh(A, max_iterations, tolerance)
    n = size(A, 1);
    x = ones(n, 1);
    x = x / norm(x);  % Normalize using Euclidean 2-norm: ||x||_2
    
    fprintf('\n>> POWER METHOD ITERATION PROGRESS:\n\n');
    fprintf('  Iteration  |  ||x_(k+1) - x_k||_2  |   Status\n');
    fprintf('  -----------+----------------------+---------------------\n');
    
    for i = 1:max_iterations
        Ax = A * x;
        x_new = Ax / norm(Ax);  % New approximation
        change = norm(x_new - x);
        
        if i <= 5 || mod(i, 10) == 0 || change < tolerance
            if change < tolerance
                status = '-> CONVERGED [OK]';
            else
                status = '';
            end
            fprintf('  %9d  |  %.10e  |  %s\n', i, change, status);
        end
        
        if change < tolerance
            fprintf('\n   [SUCCESS] Eigenvector convergence after %d iterations\n', i);
            eigenvector = x_new;
            eigenvalue = rayleighQuotient(A, eigenvector);
            return;
        end
        x = x_new;
    end
    % If convergence not achieved
    fprintf('\n   [WARNING] Maximum iterations (%d) reached\n', max_iterations);
    eigenvector = x;
    eigenvalue = rayleighQuotient(A, eigenvector);
end

function lambda = rayleighQuotient(A, x)
    x_transpose = x';                    
    numerator = x_transpose * A * x;    
    denominator = x_transpose * x;
    % Rayleigh quotient formula
    lambda = numerator / denominator;
end