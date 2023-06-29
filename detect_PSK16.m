function [est_X, est_bit_seq] = detect_PSK16(Y)
    % PSK16 Constellation points
    constellation = exp(1i * (0:15) * 2*pi/16);
    
    % Estimate the symbols using nearest neighbor rule
    est_X = zeros(length(Y), 1);
    for i = 1:length(Y)
        [~, index] = min(abs(Y(i) - constellation));
        est_X(i) = constellation(index);
    end
    
    % Gray mapping
    gray_map = [0 1 3 2 7 6 4 5 15 14 12 13 8 9 11 10]; % Gray mapping table
    
    % Convert symbols to bit sequences
    est_bit_seq = zeros(1 ,length(est_X) * 4);
    for i = 1:length(est_X)
        [~, index] = min(abs(est_X(i) - constellation));
        gray_index = gray_map(index + 1); % Add 1 to convert index to 1-based
        est_bit_seq((i-1)*4 + 1:i*4) = de2bi(gray_index, 4, 'left-msb');
    end
end