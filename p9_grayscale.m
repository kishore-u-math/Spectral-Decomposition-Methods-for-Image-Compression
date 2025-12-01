clc; clear all;
% --- Step 1: Read and convert image to grayscale ---
A_color = imread('Admin_Building.png');
A = rgb2gray(A_color);
% --- Step 2: Convert grayscale image to double precision ---
A_gray = double(A);
% --- Step 3: Get image dimensions and display information ---
[m, n] = size(A);
fprintf('\n================================================================\n');
fprintf('        IMAGE COMPRESSION USING SVD (GRAYSCALE)                 \n');
fprintf('================================================================\n\n');
fprintf('>> Image Information:\n');
fprintf('   * Dimensions: %d x %d pixels\n', m, n);
fprintf('   * Total pixels: %s\n', format_number(m*n));
fprintf('   * Channels: 1 (Grayscale)\n\n');
% --- Input for compression level ---
k = input('>> Enter the final k value for the compressed image: ');
if k > min(m, n), k = min(m, n);  % Validate k
    fprintf('   [NOTE] k adjusted to maximum: %d\n\n', k);
end
fprintf('\n>> Processing Bidiagonalization + SVD compression...\n');

%% ========== MODIFIED SECTION 4: BIDIAGONALIZATION + SVD ==========
fprintf('   * Step 1: Bidiagonalizing the image matrix...\n');
[U_bidi, B_bidi, V_bidi] = bidiagonalize(A_gray);
fprintf('   * Bidiagonal form obtained: %dx%d\n', size(B_bidi, 1), size(B_bidi, 2));

fprintf('   * Step 2: Computing SVD of bidiagonal matrix...\n');
[U_svd, S_svd, V_svd] = my_svd(B_bidi);

fprintf('   * Step 3: Combining transformations...\n');
U = U_bidi * U_svd;
V = V_bidi * V_svd;
S = S_svd;
fprintf('   * Transformation complete.\n\n');

%% --- Step 5: Create compressed image for the chosen k ---
U_k = U(:, 1:k);
S_k = S(1:k, 1:k);
V_k = V(:, 1:k);
A_compressed_double = U_k * S_k * V_k';
% Clip values to valid range [0, 255]
A_compressed_double = max(0, min(255, A_compressed_double));
A_compressed = uint8(A_compressed_double);
%% --- Step 6: Display compression statistics for the chosen k ---
fprintf('>> Compression completed for k = %d!\n\n', k);
fprintf('================================================================\n');
fprintf('                    COMPRESSION RESULTS (for k=%d)              \n', k);
fprintf('================================================================\n\n');
cr_percent = (1 - (k*(m+n+1))/(m*n))*100;
compression_ratio = (m*n) / (k*(m+n+1));
fprintf('>> Storage Analysis:\n');
fprintf('   * Original size:       %s values\n', format_number(m*n));
fprintf('   * Compressed size:     %s values\n', format_number(k*(m+n+1)));
fprintf('>> Compression Metrics:\n');
fprintf('   * Size reduction:      %.2f%%\n', cr_percent);
fprintf('   * Compression ratio:   %.2f:1\n\n', compression_ratio);
%% --- Step 7: Visualize original vs final compressed image ---
figure('Name', 'SVD Grayscale Image Compression', 'Position', [100, 100, 1200, 500]);
subplot(1, 2, 1);
imshow(A);
title(sprintf('Original Grayscale Image\n(%s values)', format_number(m*n)), 'FontSize', 12, 'FontWeight', 'bold');
subplot(1, 2, 2);
imshow(A_compressed);
title(sprintf('Compressed Image (k = %d)\n(%.2f%% reduction)', k, cr_percent), 'FontSize', 12, 'FontWeight', 'bold');
%% --- Step 8: Calculate and Plot Reconstruction Error vs. k ---
max_k_for_plot = 200;
k_values = 1:1:min(m, max_k_for_plot);
errors = zeros(length(k_values), 1);
for i = 1:length(k_values)
current_k = k_values(i);% Reconstruct image for the current k
A_k = U(:,1:current_k) * S(1:current_k,1:current_k) * V(:,1:current_k)';
errors(i) = norm(A_gray - A_k, 'fro');
end
figure('Name', 'Reconstruction Error vs. Number of Singular Values');
plot(k_values, errors, 'LineWidth', 1.5);
grid on;
title('Error in Compression', 'FontSize', 14);
xlabel('Number of Singular Values Used (k)', 'FontSize', 12);
ylabel('Error between compress and original image', 'FontSize', 12);
set(gca, 'YScale', 'linear');
%% --- Step 9: Calculate final image quality metrics for the chosen k ---
% FIXED: Use the correct variable names
A_orig_double = A_gray;
A_comp_double = A_compressed_double;
error_matrix = A_orig_double - A_comp_double;
mse_val = mean(error_matrix(:).^2);
frobenius_error = norm(error_matrix, 'fro');
MAX_VAL = 255;
if mse_val == 0
    psnr_val = inf;
    fprintf('>> PSNR is INFINITE (perfect reconstruction - MSE = 0)\n\n');
else
    psnr_val = 10 * log10((MAX_VAL^2) / mse_val);
end
fprintf('================================================================\n');
fprintf('                      QUALITY METRICS (for k=%d)                \n', k);
fprintf('================================================================\n\n');
fprintf('>> Image Quality Measurements:\n');
fprintf('   * Frobenius Norm Error: %.6e\n', frobenius_error);
fprintf('   * Mean Squared Error (MSE):  %.4f\n', mse_val);
fprintf('   * Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
if psnr_val >= 40, fprintf('   * Quality Assessment: Visually Excellent (Almost Indistinguishable)\n\n');
elseif psnr_val >= 30, fprintf('   * Quality Assessment: Good Quality\n\n');
elseif psnr_val >= 20, fprintf('   * Quality Assessment: Fair\n\n'); 
else fprintf('   * Quality Assessment: Low quality\n\n'); end
fprintf('================================================================\n\n');
%% --- User Defined Functions ---

%% ========== BIDIAGONALIZATION FUNCTION ==========
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

%% ========== HOUSEHOLDER VECTOR FUNCTION ==========
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

function [U, S, V] = my_svd(A)
    [m, n] = size(A);
    
    if m >= n
        % Case 1: m >= n (tall or square matrix)
        ATA = A' * A;
        [V_temp, D] = eig(ATA);
        [d, idx] = sort(diag(D), 'descend');
        V = V_temp(:, idx);
        
        sigma = sqrt(max(0, d));
        S = zeros(m, n);
        S(1:n, 1:n) = diag(sigma);
        
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
        % Case 2: m < n (wide matrix)
        AAT = A * A';
        [U_temp, D] = eig(AAT);
        [d, idx] = sort(diag(D), 'descend');
        U = U_temp(:, idx);
        
        sigma = sqrt(max(0, d));
        S = zeros(m, n);
        S(1:m, 1:m) = diag(sigma);
        
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
function str = format_number(num)
    str = sprintf('%d', round(num));
    k = length(str);
    
    if k > 3
        rem = mod(k, 3);
        if rem == 0
            rem = 3;
        end
        str = [str(1:rem), sprintf(',%s', reshape(str(rem+1:end), 3, [])')];
    end
end
