function w = ridge(X, y, lambda)
%RIDGE Ridge Regression.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE


X = [ones(1,size(X,2));X];  %Notice here!!!!
[P,N] = size(X);
%{
A = X*X'+lambda.*ones(P,P);
[u,s,v] = svd(A);
m = ones(P,P);
m(s==0) = 0;
s(s==0) = 1;
s = m./s;
w = v*s*u'*(X*y');
%}
if(lambda == 0)
    w = pinv(X*X'+lambda.*eye(P))*(X*y');
else
    w = (X*X'+lambda.*eye(P))\(X*y');
end
end
