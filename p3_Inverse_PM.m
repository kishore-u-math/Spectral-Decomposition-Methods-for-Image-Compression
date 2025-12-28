clc; clear all;
fprintf('\n================================================================\n');
fprintf('        INVERSE POWER METHOD ALGORITHM             \n');
fprintf('         Finding the Smallest Eigenvalue                    \n');
fprintf('================================================================\n\n');
A = [5 -2; -2 8];
fprintf('>> Input Matrix A:\n\n');
disp(A);
tolerance = 1e-7;
max_iterations = 30;

fprintf('\n>> Algorithm Parameters:\n');
fprintf('   * Maximum iterations: %d\n', max_iterations);
fprintf('   * Convergence tolerance: %.0e\n\n', tolerance);

[lambda, x, converged] = inversePowerMethod(A, tolerance, max_iterations);

if converged
    fprintf('\n================================================================\n');
    fprintf('                      FINAL RESULTS                            \n');
    fprintf('================================================================\n\n');
    
    fprintf('>> Approximated Smallest Eigenvalue (lambda):\n');
    fprintf('   lambda = %.10f\n\n', lambda);
    
    fprintf('>> Corresponding Eigenvector (x):\n');
    fprintf('   x = [');
    fprintf('%.7f', x(1));
    for i = 2:length(x)
        fprintf('; %.7f', x(i));
    end
    fprintf(']\n\n');
    all_eigenvalues = sort(eig(A)); 
    
    fprintf('>> COMPARISON WITH MATLAB eig():\n');
    fprintf('   - Inverse Power Method: %.10f\n', lambda);
    fprintf('   - MATLAB''s Smallest Eig: %.10f\n', all_eigenvalues(1));
    fprintf('   - Absolute Difference:  %.6e\n\n', abs(lambda - all_eigenvalues(1)));
end
function [lambda, x, success] = inversePowerMethod(A, TOL, N)    
    [n, ~] = size(A);
    success = 0; 
    x0 = ones(n, 1);
    x = x0 / norm(x0);    
    fprintf('>> INVERSE POWER METHOD ITERATION PROGRESS:\n\n');
    fprintf('  Iteration  |  Eigenvalue Estimate  |  ||x_(k+1) - x_k||_2  |   Status\n');
    fprintf('  -----------+-----------------------+---------------------+----------------\n');
    for k = 1:N
        if rcond(A) < 1e-12
            fprintf('\n   [ERROR] Matrix A is singular or ill-conditioned.\n');
            lambda = 0;
            return;
        end
        y = A \ x;
        norm_y = norm(y);
        lambda = 1 / norm_y;
        x_new = y / norm_y;
        err = norm(x_new - x);
        status = '';
        if err < TOL, status = '-> CONVERGED [OK]'; end
        fprintf('  %9d  |  %19.10f  |  %19.6e  | %s\n', k, lambda, err, status);
        if err < TOL
            fprintf('\n   [SUCCESS] Convergence reached after %d iterations.\n', k);
            x = x_new;
            success = 1;
            return; 
        end
        x = x_new;
    end
    fprintf('\n   [WARNING] Method did not converge within %d iterations.\n', N);
    lambda = 1 / norm(y);
    x = y / norm(y);      
end