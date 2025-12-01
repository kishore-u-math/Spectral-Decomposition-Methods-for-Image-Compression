clc; clear all;

%% ========================================================================
%         IMAGE COMPRESSION USING SVD DECOMPOSITION (COLOR)
%% ========================================================================

% --- Step 1: Read color image ---
A = imread('Admin_Building.png');  

% --- Step 2: Extract and convert RGB channels to double precision ---
R = double(A(:,:,1));
G = double(A(:,:,2));
B = double(A(:,:,3));

% --- Step 3: Get image dimensions and display information ---
[m, n, channels] = size(A);
fprintf('\n================================================================\n');
fprintf('        IMAGE COMPRESSION USING SVD DECOMPOSITION               \n');
fprintf('================================================================\n\n');
fprintf('>> Image Information:\n');
fprintf('   * Dimensions: %d x %d pixels\n', m, n);
fprintf('   * Original data points: %s values\n\n', format_number(m*n*channels));

% --- User input for compression level ---
k = input('>> Enter the final k value for the compressed image: ');

if k > min(m, n), k = min(m, n);
    fprintf('   [NOTE] k adjusted to maximum: %d\n\n', k);
end

fprintf('\n>> Processing SVD compression...\n');

%% --- Step 4: Apply Full SVD to each channel ---
[U_R, S_R, V_R] = my_svd(R);
[U_G, S_G, V_G] = my_svd(G);
[U_B, S_B, V_B] = my_svd(B);

%% --- Step 5: Create compressed image for the chosen k ---
R_compressed = U_R(:,1:k) * S_R(1:k,1:k) * V_R(:,1:k)';
G_compressed = U_G(:,1:k) * S_G(1:k,1:k) * V_G(:,1:k)';
B_compressed = U_B(:,1:k) * S_B(1:k,1:k) * V_B(:,1:k)';
A_compressed = uint8(cat(3, R_compressed, G_compressed, B_compressed));

%% --- Step 6: Display compression statistics for the chosen k ---
fprintf('>> Compression completed for k = %d\n\n', k);
fprintf('================================================================\n');
fprintf('                    COMPRESSION RESULTS (for k=%d)              \n', k);
fprintf('================================================================\n\n');
cr_percent = (1 - (k*(m+n+1))/(m*n))*100;
fprintf('>> Storage Analysis (per channel):\n');
fprintf('   * Original size:       %s values\n', format_number(m*n*3));
fprintf('   * Compressed size:     %s values\n', format_number(3*k*(m+n+1)));
fprintf('>> Compression Metrics:\n');
fprintf('   * Size reduction:      %.2f%%\n\n', cr_percent);

%% --- Step 7: Visualize original vs final compressed image ---
figure('Name', 'SVD Color Image Compression', 'Position', [100, 100, 1200, 500]);
subplot(1, 3, 1);
imshow(A);
title(sprintf('Original Image\n(%s values)', format_number(m*n*channels)), 'FontSize', 12);
subplot(1, 3, 2);
imshow(A_compressed);
title(sprintf('Compressed Image (Top %d Singular Values)\n(%.2f%% reduction)', k, cr_percent), 'FontSize', 12);
subplot(1, 3, 3);
imshow(A-A_compressed);
title(sprintf('Removed Pixels from Original Image \n (Removed %d Singular Values)', 600-k), 'FontSize', 12);

%% --- Step 8: Calculate and Plot Reconstruction Error for each channel ---
max_k_for_plot = 200;
k_values = 1:5:min(m, max_k_for_plot); % Use steps to make it faster
errors_R = zeros(length(k_values), 1);
errors_G = zeros(length(k_values), 1);
errors_B = zeros(length(k_values), 1);

for i = 1:length(k_values)
    current_k = k_values(i);
    % Reconstruct each channel for the current k
    R_k = U_R(:,1:current_k) * S_R(1:current_k,1:current_k) * V_R(:,1:current_k)';
    G_k = U_G(:,1:current_k) * S_G(1:current_k,1:current_k) * V_G(:,1:current_k)';
    B_k = U_B(:,1:current_k) * S_B(1:current_k,1:current_k) * V_B(:,1:current_k)';
    
    % Calculate Frobenius norm error for each channel
    errors_R(i) = norm(R - R_k, 'fro');
    errors_G(i) = norm(G - G_k, 'fro');
    errors_B(i) = norm(B - B_k, 'fro');
end

% Plotting the error graph
figure('Name', 'Reconstruction Error per Channel');
hold on;
plot(k_values, errors_R, 'r', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
plot(k_values, errors_G, 'G', 'LineWidth', 1.5, 'MarkerFaceColor', 'g');
plot(k_values, errors_B, 'b', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold off;

grid on;
title('Error in compression', 'FontSize', 14);
xlabel('Number of Singular Values used', 'FontSize', 12);
ylabel('Error between compress and original image', 'FontSize', 12);
legend('Red Channel', 'Green Channel', 'Blue Channel', 'Location', 'northeast');
set(gca, 'YScale', 'linear');

%% --- Step 9: Calculate final image quality metrics for the chosen k ---
A_orig_double = double(A);
A_comp_double = double(cat(3,R_compressed,G_compressed,B_compressed));
error_matrix = A_orig_double - A_comp_double;
mse_val = mean(error_matrix(:).^2);

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
fprintf('>> Overall Image Quality Measurements:\n');
fprintf('   * Frobenius Norm Error: %.6e\n', norm(error_matrix, 'fro'));
fprintf('   * Mean Squared Error (MSE):  %.4f\n', mse_val);
fprintf('   * Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
if psnr_val >= 40, fprintf('   * Quality Assessment: Visually Excellent (Almost Indistinguishable)\n\n');
elseif psnr_val >= 30, fprintf('   * Quality Assessment: Good Quality\n\n');
elseif psnr_val >= 20, fprintf('   * Quality Assessment: Fair\n\n'); 
else fprintf('   * Quality Assessment: Low quality\n\n'); end

fprintf('================================================================\n\n');

% --- CUSTOMISED SINGULAR VALUE DECOMPOSITION ALGORITHM ---

function [U, S, V] = my_svd(A)
    [m, n] = size(A);
    if m >= n
        ATA = A' * A; [V_temp, D] = eig(ATA);
        [d, idx] = sort(diag(D), 'descend'); V = V_temp(:, idx);
        sigma = sqrt(max(0, d)); S = zeros(m, n); S(1:n, 1:n) = diag(sigma);
        U_partial = zeros(m, n);
        for i=1:n, if sigma(i)>1e-10, U_partial(:,i)=(A*V(:,i))/sigma(i); end, end
        if m>n, U=complete_basis_gramschmidt(U_partial,m); else, U=U_partial; end
    else
        AAT = A*A'; [U_temp, D] = eig(AAT);
        [d, idx] = sort(diag(D), 'descend'); U = U_temp(:, idx);
        sigma = sqrt(max(0, d)); S = zeros(m, n); S(1:m, 1:m) = diag(sigma);
        V_partial = zeros(n, m);
        for i=1:m, if sigma(i)>1e-10, V_partial(:,i)=(A'*U(:,i))/sigma(i); end, end
        if n>m, V=complete_basis_gramschmidt(V_partial,n); else, V=V_partial; end
    end
end
function Q = complete_basis_gramschmidt(Q_partial, n)
    [m, k] = size(Q_partial); Q = [Q_partial, zeros(m, n-k)];
    for j = k+1:n
        v = randn(m, 1);
        for i=1:j-1, v=v-(Q(:,i)'*v)*Q(:,i); end
        v_norm = norm(v); attempts=0;
        while v_norm<1e-10 && attempts<10
            v=randn(m,1);
            for i=1:j-1, v=v-(Q(:,i)'*v)*Q(:,i); end
            v_norm=norm(v); attempts=attempts+1;
        end
        Q(:,j) = v/v_norm;
    end
end
function str = format_number(num)
    str=sprintf('%d',round(num)); k=length(str);
    if k>3, rem=mod(k,3); if rem==0, rem=3; end; str=[str(1:rem),sprintf(',%s',reshape(str(rem+1:end),3,[])')]; end
end