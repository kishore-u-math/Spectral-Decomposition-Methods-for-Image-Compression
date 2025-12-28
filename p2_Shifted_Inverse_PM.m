clc; clear all;
fprintf('\n================================================================\n');
fprintf('        SHIFTED INVERSE POWER METHOD ALGORITHM             \n');
fprintf('       Finding an Eigenvalue and Eigenvector                    \n');
fprintf('================================================================\n\n');
A = [ 80    72     40;   72   170    140;   40   140    200 ];
fprintf('>> Input Matrix A:\n\n');
disp(A);

% Define an initial guess for the eigenvector
x_initial = [1; 1; 1];
fprintf('>> Initial Eigenvector Guess (x0):\n\n');
disp(x_initial);
max_iter = 20;       % Maximum number of iterations
tol = 1e-6;          % Convergence tolerance

fprintf('\n>> Algorithm Parameters:\n');
fprintf('   * Maximum iterations: %d\n', max_iter);
fprintf('   * Convergence tolerance: %.0e\n\n', tol);
[eigenvalue, eigenvector, converged] = ...
    inversePowerMethod(A, x_initial, tol, max_iter);
if converged
    fprintf('\n================================================================\n');
    fprintf('                      FINAL RESULTS                            \n');
    fprintf('================================================================\n\n');
    
    fprintf('>> Approximated Eigenvalue (mu):\n');
    fprintf('   mu = %.10f\n\n', eigenvalue);
    
    fprintf('>> Approximated Eigenvector (x):\n');
    fprintf('   x = [');
    fprintf('%.7f', eigenvector(1));
    for i = 2:length(eigenvector)
        fprintf('; %.7f', eigenvector(i));
    end
    fprintf(']\n\n');
    residual = A * eigenvector - eigenvalue * eigenvector;
    residual_norm = norm(residual);
    fprintf('>> VERIFICATION (A*x = mu*x):\n');
    fprintf('   * ||A*x - mu*x|| = %.6e\n\n', residual_norm);

    % --- Verification using MATLAB's built-in eig() function ---
    [eig_vecs, eig_vals_diag] = eig(A);
    eig_vals = diag(eig_vals_diag);    
    [~, min_idx] = min(abs(eig_vals - eigenvalue));    
    fprintf('>> COMPARISON WITH MATLAB eig():\n');
    fprintf('   - Shifted Inverse Power Method: %.10f\n', eigenvalue);
    fprintf('   - MATLAB''s Closest Eig: %.10f\n', eig_vals(min_idx));
    fprintf('   - Absolute Difference:  %.6e\n\n', abs(eigenvalue - eig_vals(min_idx)));
end
function [mu, x, success] = inversePowerMethod(A, x_init, TOL, N)
    success = 0;
    n = size(A, 1);
    q = (x_init' * A * x_init) / (x_init' * x_init);
    fprintf('>> Initial shift `q` (from Rayleigh Quotient) = %.6f\n', q);
    x = x_init;
    [~, p] = max(abs(x)); 
    x = x / x(p);             
    fprintf('\n>> SHIFTED INVERSE POWER METHOD ITERATION PROGRESS:\n\n');
    fprintf('  Iteration  |  Eigenvalue Approx  |  Error (inf-norm)   |   Status\n');
    fprintf('  -----------+---------------------+-----------------------+----------------\n');
    for k = 1:N
        M = A - q * eye(n);
        if rcond(M) < 1e-12
            fprintf('\n   [ERROR] Matrix (A - qI) is singular or ill-conditioned.\n');
            fprintf('   The shift q = %.6f is likely an exact eigenvalue.\n', q);
            mu = q;
            success = 1;
            return;
        end
        y = M \ x;
        [~, p_y] = max(abs(y));
        mu_intermediate = y(p_y);
        y_normalized = y / mu_intermediate; 
        ERR = norm(x - y_normalized, inf);  
        x = y_normalized;
        current_eigenvalue_approx = (1 / mu_intermediate) + q;
        status = '';
        if ERR < TOL, status = '-> CONVERGED [OK]'; end
        fprintf('  %9d  |  %17.10f  |  %19.6e  | %s\n', k, current_eigenvalue_approx, ERR, status);
        if ERR < TOL
            mu = current_eigenvalue_approx;
            fprintf('\n   [SUCCESS] Convergence reached after %d iterations.\n', k);
            success = 1;
            return;
        end
    end
    fprintf('\n   [WARNING] Maximum iterations (%d) exceeded.\n', N);
    mu = (1 / mu_intermediate) + q;
end
