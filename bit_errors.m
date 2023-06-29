function num_bit_errors = bit_errors(est_bit_seq, b)
    % Calculate the number of bit errors
    num_bit_errors = sum(est_bit_seq ~= b);
end