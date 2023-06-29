function [X, Xi,Xq] = bits_to_PSK16(b_seq)
    N = length(b_seq);
    X = zeros(N/4, 1); % Initialize the vector X with zeros
    counter = 1; % Initialize the counter

   

    % Iterate over each sequence of 4 bits
    for k = 1:4:N-3
        % Check for each possible combination of the 4 bits
        if b_seq(k) == 0 && b_seq(k+1) == 0 && b_seq(k+2) == 0 && b_seq(k+3) == 0
            X(counter) = 1;
        elseif b_seq(k) == 1 && b_seq(k+1) == 0 && b_seq(k+2) == 0 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*1/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 0 && b_seq(k+2) == 0 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*2/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 0 && b_seq(k+2) == 1 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*3/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 0 && b_seq(k+2) == 1 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*4/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 1 && b_seq(k+2) == 1 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*5/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 1 && b_seq(k+2) == 1 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*6/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 1 && b_seq(k+2) == 0 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*7/16);
        elseif b_seq(k) == 1 && b_seq(k+1) == 1 && b_seq(k+2) == 0 && b_seq(k+3) == 0
            X(counter) = -1;
        elseif b_seq(k) == 0 && b_seq(k+1) == 1 && b_seq(k+2) == 0 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*9/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 1 && b_seq(k+2) == 0 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*10/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 1 && b_seq(k+2) == 1 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*11/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 1 && b_seq(k+2) == 1 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*12/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 0 && b_seq(k+2) == 1 && b_seq(k+3) == 0
            X(counter) = exp(2*pi*1i*13/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 0 && b_seq(k+2) == 1 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*14/16);
        elseif b_seq(k) == 0 && b_seq(k+1) == 0 && b_seq(k+2) == 0 && b_seq(k+3) == 1
            X(counter) = exp(2*pi*1i*15/16);
        end

        counter = counter + 1;
    end
    Xi=real(X);
    Xq=imag(X);
end
