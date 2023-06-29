function num_symbol_errors = symbol_errors(est_X, X)
    % Calculate the number of symbol errors
    num_symbol_errors = sum(est_X ~= X);
end