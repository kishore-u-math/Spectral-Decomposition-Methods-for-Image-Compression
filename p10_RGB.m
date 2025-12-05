clc; clear all;

%% ========================================================================
%         IMAGE COMPRESSION USING SVD DECOMPOSITION (COLOR)
%% ========================================================================

% --- Step 1: Read color image ---
A = imread('Admin_Building.png');  %Admin_Building.png  %sbs_main.jpeg  %sbs.jpg 

% --- Extract and convert RGB channels to double precision ---
R = double(A(:,:,1));
G = double(A(:,:,2));
B = double(A(:,:,3));

% --- Create colored channel images ---
RedImage   = cat(3, R, zeros(size(R), 'uint8'), zeros(size(R), 'uint8'));
GreenImage = cat(3, zeros(size(G), 'uint8'), G, zeros(size(G), 'uint8'));
BlueImage  = cat(3, zeros(size(B), 'uint8'), zeros(size(B), 'uint8'), B);

% --- Step 2: Display RGB Image ---
figure('Name', 'SVD Color Image - Channel Display', 'Position', [100, 100, 1200, 500]);
% Column 1: Red Channel
subplot(1,4,1);
imshow(RedImage);
title('Red Channel', 'FontSize', 11, 'FontWeight', 'bold');

% Column 2: Green Channel
subplot(1,4,2);
imshow(GreenImage);
title('Green Channel', 'FontSize', 11, 'FontWeight', 'bold');

% Column 3: Blue Channel
subplot(1,4,3);
imshow(BlueImage);
title('Blue Channel', 'FontSize', 11, 'FontWeight', 'bold');

% Column 4: Original image
subplot(1,4,4);
imshow(A);
title('Original Image', 'FontSize', 11, 'FontWeight', 'bold');

%% --- Step 3: Get image dimensions and display information ---
[m, n, channels] = size(A);
fprintf('\n================================================================\n');
fprintf('        IMAGE COMPRESSION USING SVD DECOMPOSITION               \n');
fprintf('================================================================\n\n');
fprintf('>> Image Information:\n');
fprintf('   * Dimensions (m x n)           : %d x %d pixels\n', m, n);
fprintf('   * No. of Channels              : %d (RGB)\n',channels);
fprintf('   * Total No. of Singular Values : %d \n', min(m,n));
fprintf('   * Original data points(3mn)    : %s values\n\n', format_number(m*n*channels));

%% --- Step 4: Apply Full SVD to each channel ---
%k = input('>> Enter the k value for compressing image: '); %User input for compression level
k=25;
if k > min(m, n), k = min(m, n)/2;
    fprintf('   [NOTE] k adjusted to half the maximum: %d\n\n', k);
end

fprintf('\n>> Processing SVD compression...\n');
[U_R, S_R, V_R] = my_svd(R);
[U_G, S_G, V_G] = my_svd(G);
[U_B, S_B, V_B] = my_svd(B);

%% --- Step 5: Create compressed image for the given k ---
R_compressed = U_R(:,1:k) * S_R(1:k,1:k) * V_R(:,1:k)';
G_compressed = U_G(:,1:k) * S_G(1:k,1:k) * V_G(:,1:k)';
B_compressed = U_B(:,1:k) * S_B(1:k,1:k) * V_B(:,1:k)';
A_compressed = uint8(cat(3, R_compressed, G_compressed, B_compressed));
fprintf('>> Compression completed for k = %d\n\n', k);

%% --- Step 6: Visualize original vs final compressed image ---
figure('Name', 'SVD Color Image Compression', 'Position', [100, 100, 1200, 500]);
subplot(1, 3, 1);
imshow(A);
title(sprintf('Original Image\n(%s values)', format_number(m*n*channels)), 'FontSize', 12);
subplot(1, 3, 2);
imshow(A_compressed);
cr_percent = (1 - (k*(m+n+1))/(m*n))*100;
title(sprintf('Compressed Image (Top %d Singular Values)\n(%.2f%% reduction)', k, cr_percent), 'FontSize', 12);
subplot(1, 3, 3);
imshow(A-A_compressed);
title(sprintf('Removed Pixels from Original Image \n (Removed %d Singular Values)', min(m,n)-k), 'FontSize', 12);
%% --- Step 7: Display compression statistics for the chosen k ---
fprintf('================================================================\n');
fprintf('                    COMPRESSION RESULTS (for k=%d)              \n', k);
fprintf('================================================================\n\n');
fprintf('>> Storage Analysis :\n');
fprintf('   * Original size (3mn)        : %s values\n', format_number(m*n*3));
fprintf('   * Compressed size (3k(m+n+1)): %s values\n', format_number(3*k*(m+n+1)));
fprintf('>> Compression Metrics:\n');
fprintf('   * Size reduction             : %.2f%%\n\n', cr_percent);

%% --- Step 8: Calculate final image quality metrics for the chosen k ---
A_orig_double = double(A);
A_comp_double = double(cat(3,R_compressed,G_compressed,B_compressed));
error_matrix = A_orig_double - A_comp_double;
mse_val = mean(error_matrix(:).^2);

MAX_VAL = 255;
if mse_val == 0
    psnr_val = inf;
    fprintf('>> PSNR is INFINITE (Perfect Reconstruction)\n\n');
else
    psnr_val = 10 * log10((MAX_VAL^2) / mse_val);
end

fprintf('================================================================\n');
fprintf('                      QUALITY METRICS (for k=%d)                \n', k);
fprintf('================================================================\n\n');
fprintf('>> Overall Image Quality Measurements:\n');
fprintf('   * Frobenius Norm Error             : %.6e\n', norm(error_matrix, 'fro'));
fprintf('   * Mean Squared Error (MSE)         :  %.4f\n', mse_val);
fprintf('   * Peak Signal-to-Noise Ratio (PSNR): %.2f dB\n', psnr_val);
if psnr_val >= 40, fprintf('   * Quality Assessment: Visually Excellent (Almost Indistinguishable)\n\n');
elseif psnr_val >= 30, fprintf('   * Quality Assessment               : Good Quality\n\n');
elseif psnr_val >= 20, fprintf('   * Quality Assessment               : Fair\n\n'); 
else fprintf('   * Quality Assessment               : Low quality\n\n'); end

fprintf('================================================================\n\n');

%% --- Step 9a: Calculate and Plot Reconstruction Error(Frobenius)for each channel ---
max_k_for_plot = 200;  s=min(m,n);
k_values = 1:5:min(s, max_k_for_plot);
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
figure('Name', 'Reconstruction Error per Channel','Position', [100, 100, 1200, 500]);
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

%% --- Step 9b: Calculate and Plot PSNR for different k values ---
fprintf('>> Calculating PSNR for different k values...\n\n');

max_k_for_psnr = 200;   s=min(m,n);
k_values_psnr = 1:5:min(s, max_k_for_psnr);
psnr_values = zeros(length(k_values_psnr), 1);

% Note: A_orig_double needs to be calculated here
A_orig_double = double(A);
MAX_VAL = 255;

for i = 1:length(k_values_psnr)
    current_k = k_values_psnr(i);
    
    % Reconstruct each channel for the current k
    R_k = U_R(:,1:current_k) * S_R(1:current_k,1:current_k) * V_R(:,1:current_k)';
    G_k = U_G(:,1:current_k) * S_G(1:current_k,1:current_k) * V_G(:,1:current_k)';
    B_k = U_B(:,1:current_k) * S_B(1:current_k,1:current_k) * V_B(:,1:current_k)';
    
    % Combine channels and calculate PSNR
    A_comp_double_k = cat(3, R_k, G_k, B_k);
    error_matrix_k = A_orig_double - A_comp_double_k;
    mse_val_k = mean(error_matrix_k(:).^2);
    
    if mse_val_k == 0
        psnr_values(i) = inf;
    else
        psnr_values(i) = 10 * log10((MAX_VAL^2) / mse_val_k);
    end
end

% Plotting the PSNR graph
figure('Name', 'PSNR vs Number of Singular Values','Position', [100, 100, 1200, 500]);
plot(k_values_psnr, psnr_values, 'G-o', 'LineWidth', 2, 'MarkerSize', 4);
grid on;
title('Peak Signal-to-Noise Ratio (PSNR) vs Number of Singular Values', 'FontSize', 14);
xlabel('Number of Singular Values (k)', 'FontSize', 12);
ylabel('PSNR (dB)', 'FontSize', 12);

% Add reference line at PSNR = 40 dB
hold on;
yline(40, 'r--', 'LineWidth', 1.5);
text(max(k_values_psnr)*0.7, 42, 'PSNR = 40 dB (Excellent Quality)', 'Color', 'r', 'FontSize', 10);
hold off;

% Find the minimum k value where PSNR >= 40 dB
idx_40db = find(psnr_values >= 40, 1, 'first'); %return 1st index

fprintf('================================================================\n');
fprintf('           OPTIMAL k VALUE FOR PSNR >= 40 dB                    \n');
fprintf('================================================================\n\n');
if ~isempty(idx_40db)
    k_optimal = k_values_psnr(idx_40db);
    psnr_optimal = psnr_values(idx_40db);
    fprintf('>> Minimum k value for PSNR >= 40 dB:\n');
    fprintf('   * Optimal k               :%d\n', k_optimal);
    fprintf('   * PSNR at this k          :%.2f dB\n', psnr_optimal);
    fprintf('   * Quality Assessment      :Visually Excellent (Almost Indistinguishable)\n');
    % Calculate compression for optimal k
    cr_optimal = (1 - (k_optimal*(m+n+1))/(m*n))*100;
    fprintf('   * Size reduction          :%.2f%%\n', cr_optimal);
else
    fprintf('>> No k value in the tested range achieves PSNR >= 40 dB\n');
    fprintf('   * Maximum PSNR achieved:   %.2f dB at k = %d\n', max(psnr_values), k_values_psnr(psnr_values == max(psnr_values)));
    fprintf('   * Consider increasing max_k_for_psnr if needed\n\n');
end

fprintf('================================================================\n\n');


%% --- CUSTOMISED SINGULAR VALUE DECOMPOSITION ALGORITHM ---

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
    [m, k] = size(Q_partial); Q = [Q_partial, zeros(m, n-k)];%concatate
    for j = k+1:n
        v = randn(m, 1); %random column vector
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
    str=sprintf('%d',round(num)); k=length(str); %Convert number to string & index
    if k>3, rem=mod(k,3); if rem==0, rem=3; end; str=[str(1:rem),sprintf(',%s',reshape(str(rem+1:end),3,[])')]; end

end
