% You can use this skeleton or write your own.
% You are __STRONGLY__ suggest to run this script section-by-section using Ctrl+Enter.
% See http://www.mathworks.cn/cn/help/matlab/matlab_prog/run-sections-of-programs.html for more details.

%% Part1: Preceptron
fprintf('Perception:\n');
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 1000;

iter_total = 0;
training_error = 0;
testing_error = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    [w_g, iter] = perceptron(X, y);
    % Compute training, testing error
    % Sum up number of iterations
    
    %Compute training error
    training_res = w_g'*bsxfun(@times,[ones(1,size(X,2));X],y);
    training_error = training_error + sum(training_res<0)/nTrain;
    
    %Generate testing data
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_res = w_g'*bsxfun(@times,[ones(1,size(test_X,2));test_X],test_Y);
    testing_error = testing_error + sum(testing_res<0)/nTest;
    
    iter_total = iter_total+iter;
  %  plotdata(X, y, w_f, w_g, 'Pecertron');
  %  pause();
end

E_train = training_error/nRep;
E_test = testing_error/nRep;
avgIter = iter_total/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron');

%% Part2: Preceptron: Non-linearly separable case
% this part consumes large time, so I comment it. 
% You can make it running if you want.
fprintf('Non-linearly separable perceptron\n');
nTrain = 100; % number of training data
nRep = 1;
nTest = 1000;

iter_total = 0;
training_error = 0;
testing_error = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain,'noisy');
    [w_g, iter] = perceptron(X, y);
    % Compute training, testing error
    % Sum up number of iterations
    
    %Compute training error
    training_res = w_g'*bsxfun(@times,[ones(1,size(X,2));X],y);
    training_error = training_error + sum(training_res<0)/nTrain;
    
    %Generate testing data
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_res = w_g'*bsxfun(@times,[ones(1,size(test_X,2));test_X],test_Y);
    testing_error = testing_error + sum(testing_res<0)/nTest;
    
    iter_total = iter_total+iter;
  %  plotdata(X, y, w_f, w_g, 'Pecertron');
  %  pause();
end
E_train = training_error/nRep;
E_test = testing_error/nRep;
avgIter = iter_total/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
fprintf('Average number of iterations is %d.\n', avgIter);
plotdata(X, y, w_f, w_g, 'Pecertron - noisy');


%% Part3: Linear Regression
fprintf('Linear Regression:\n'); 

nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 1000; % number of testing data

E_train = 0;
E_test = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    w_g = linear_regression(X, y);
    % Compute training, testing error
    training_error = y'.*([ones(1,size(X,2));X]'*w_g);
    training_error = sum(training_error<0);
    
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_error = test_Y'.*([ones(1,size(test_X,2));test_X]'*w_g);
    testing_error = sum(testing_error<0);
    
    E_train = E_train+training_error/nTrain;
    E_test = E_test+testing_error/nTest;
end

E_train = E_train/nRep;
E_test = E_test/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression');

%% Part4: Linear Regression: noisy
fprintf('Linear Regression - noisy:\n');
nRep = 1000; % number of replicates
nTrain = 100; % number of training data
nTest = 1000; % number of testing data

E_train = 0;
E_test = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = linear_regression(X, y);
    % Compute training, testing error
    training_error = y'.*([ones(1,size(X,2));X]'*w_g);
    training_error = sum(training_error<0);
    
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_error = test_Y'.*([ones(1,size(test_X,2));test_X]'*w_g);
    testing_error = sum(testing_error<0);
    
    E_train = E_train+training_error/nTrain;
    E_test = E_test+testing_error/nTest;
end


E_train = E_train/nRep;
E_test = E_test/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Linear Regression: noisy');

%% Part5: Linear Regression: poly_fit
fprintf('Linear Regression - poly_fit:\n');
load('poly_train', 'X', 'y');
load('poly_test', 'X_test', 'y_test');
w = linear_regression(X, y)
% Compute training, testing error
training_error = y'.*([ones(1,size(X,2));X]'*w);
training_error = sum(training_error<0);

testing_error = y_test'.*([ones(1,size(X_test,2));X_test]'*w);
testing_error = sum(testing_error<0);

E_train = training_error/size(X,2);
E_test = testing_error/size(X_test,2);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


% poly_fit with transform
fprintf('poly_fit with transform\n');
X_t = [X;X(1,:).*X(2,:);X(1,:).*X(1,:);X(2,:).*X(2,:)]; % CHANGE THIS LINE TO DO TRANSFORMATION
X_test_t = [X_test;X_test(1,:).*X_test(2,:);X_test(1,:).*X_test(1,:);X_test(2,:).*X_test(2,:);]; % CHANGE THIS LINE TO DO TRANSFORMATION
w = linear_regression(X_t, y)
% Compute training, testing error
training_error = y'.*([ones(1,size(X_t,2));X_t]'*w);
training_error = sum(training_error<0);

testing_error = y_test'.*([ones(1,size(X_test_t,2));X_test_t]'*w);
testing_error = sum(testing_error<0);

E_train = training_error/size(X,2);
E_test = testing_error/size(X_test,2);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);


%% Part6: Logistic Regression
fprintf('Logistic Regression:\n');
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 100;
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    w_g = logistic(X, y);
    % Compute training, testing error
    p1 = 1./(1+exp(-w_g'*[ones(1,size(X,2));X]));
    training_error = sign(p1-0.5).*y;
    training_error = sum(training_error<0);
    
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    p1 = 1./(1+exp(-w_g'*[ones(1,size(test_X,2));test_X]));
    testing_error = sign(p1-0.5).*test_Y;
    testing_error = sum(testing_error<0);
    
    E_train = E_train + training_error/nTrain;
    E_test = E_test + testing_error/nTest;
end

E_train = E_train/nRep;
E_test = E_test/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression');

%% Part7: Logistic Regression: noisy
fprintf('Logistic Regression - noisy:\n');
nRep = 100; % number of replicates
nTrain = 100; % number of training data
nTest = 10000; % number of training data
E_train = 0;
E_test = 0;

for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain, 'noisy');
    w_g = logistic(X, y);
    % Compute training, testing error
    training_error = y'.*([ones(1,size(X,2));X]'*w_g);
    training_error = sum(training_error<0);
    
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_error = test_Y'.*([ones(1,size(test_X,2));test_X]'*w_g);
    testing_error = sum(testing_error<0);
    
    E_train = E_train + training_error/nTrain;
    E_test = E_test + testing_error/nTest;
end

E_train = E_train/nRep;
E_test = E_test/nRep;
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'Logistic Regression: noisy');

%% Part8: SVM
nRep = 1000; % number of replicates
nTrain = 10; % number of training data
nTest = 10000;

E_train = 0;
E_test = 0;
E_num = 0;
for i = 1:nRep
    [X, y, w_f] = mkdata(nTrain);
    [w_g, num_sc] = svm(X, y);
    % Compute training, testing error
    % Sum up number of support vectors
    training_error = y'.*([ones(1,size(X,2));X]'*w_g);
    training_error = sum(training_error<0);
    
    range = [-1, 1];
    dim = size(X,1);
    test_X = rand(dim, nTest)*(range(2)-range(1)) + range(1);
    test_Y = sign(w_f'*[ones(1,size(test_X,2));test_X]);
    
    %Compute testing error
    testing_error = test_Y'.*([ones(1,size(test_X,2));test_X]'*w_g);
    testing_error = sum(testing_error<0);
    
    E_train = E_train + training_error/nTrain;
    E_test = E_test + testing_error/nTest;
    E_num = E_num+num_sc;
end

E_train = E_train/nRep;
E_test = E_test/nRep;
E_num = E_num/nRep;
fprintf('E_num is %f.\n',E_num);
fprintf('E_train is %f, E_test is %f.\n', E_train, E_test);
plotdata(X, y, w_f, w_g, 'SVM');

