function [w, num] = svm(X, y)
%SVM Support vector machine.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           num:  number of support vectors
%

% YOUR CODE HERE

[P,N] = size(X);
X = [ones(1,N);X];
H = eye(P+1,P+1);
f = zeros(P+1,1);
A = -bsxfun(@times,X,y)';
b = -ones(N,1);
w = quadprog(H,f,A,b);
num = sum(abs(1-abs(w'*X))<=0.0001);
end
