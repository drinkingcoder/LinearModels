function X = feature_normalization(X_in)
    X_in_mean = mean(X_in, 2);
    X_in = bsxfun(@minus, X_in , X_in_mean);
    X_in_var = sum(X_in.^2,2)/size(X_in ,2);
    X_in_var = X_in_var + (X_in_var == 0);%avoid divide by 0
    X = bsxfun(@rdivide, X_in, sqrt(X_in_var));
end