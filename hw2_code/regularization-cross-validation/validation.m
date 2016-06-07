%% Ridge Regression
load('digit_train', 'X', 'y');
%show_digit(X);
% Do feature normalization
% ...

[P,N] = size(X);

mean = sum(X,2)./N;
X = bsxfun(@minus,X,mean);
deviation = sqrt(sum(X.^2,2)./N);
deviation(deviation==0) = 1;
X = bsxfun(@rdivide,X,deviation);

% Do LOOCV
%{
lambdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e1, 1e2, 1e3];
lambda = 0;
Min_err = N;

for i = 1:length(lambdas)
    E_val = 0
    for j = 1:size(X, 2)
        X_ = X(:,[1:j-1,j+1:N]); y_ = y(:,[1:j-1,j+1:N]); % take point j out of X
        w = ridge(X_, y_, lambdas(i));
        E_val = E_val + (w'*[1;X(:,j)]*y(1,j)<0);
    end
    % Update lambda according validation error
    if(Min_err>E_val)
        lambda = lambdas(i);
        Min_err = E_val;
    end
end
%}
lambda = 0;
w = ridge(X,y,lambda);
fprintf('lambda = %d\n',lambda);
fprintf('w=%d\n',norm(w,2)^2);
% Compute training error

training_error = y'.*([ones(1,size(X,2));X]'*w);
E_train = sum(training_error<0)/N;

load('digit_test', 'X_test', 'y_test');
% Do feature normalization

[P,N_test] = size(X_test);
mean = sum(X_test,2)./N_test;
X_test = bsxfun(@minus,X_test,mean);
deviation = sqrt(sum(X_test.^2,2)./N_test);
deviation(deviation==0) = 1;
X_test = bsxfun(@rdivide,X_test,deviation);

% Compute test error
testing_error = y_test'.*([ones(1,size(X_test,2));X_test]'*w);
E_test = sum(testing_error<0)/size(X_test,2);
fprintf('E_test = %d\n',E_test);
%% Logistic
disp('Logistic');
lambda = 0;
Min_err = N;

for i = 1:length(lambdas)
    E_val = 0
    for j = 1:size(X, 2)
        X_ = X(:,[1:j-1,j+1:N]); y_ = y(:,[1:j-1,j+1:N]); % take point j out of X
        w = logistic_r(X_, y_, lambdas(i));
        E_val = E_val + ((w'*[1;X(:,j)]*y(1,j))<0);
    end
    % Update lambda according validation error
    E_val
    if(Min_err>E_val)
        lambda = lambdas(i);
        Min_err = E_val;
    end
end

lambda = 100
w = logistic_r(X, y, lambda);
testing_error = y_test'.*([ones(1,size(X_test,2));X_test]'*w);
E_test = sum(testing_error<0)/size(X_test,2);
training_error = y'.*([ones(1,size(X,2));X]'*w);
E_train = sum(training_error<0)/size(X,2);
fprintf('E_test = %d\n',E_test);
fprintf('E_train = %d\n',E_train);
%% SVM with slack variable
[w_g, num_sc] = svm(X, y);
   
training_error = y'.*([ones(1,size(X,2));X]'*w_g);
training_error = sum(training_error<0);
E_train = training_error/size(X,2);

testing_error = y_test'.*([ones(1,size(X_test,2));X_test]'*w_g);
testing_error = sum(testing_error<0);
E_test = testing_error/size(X_test,2);

E_test
E_train
num_sc
