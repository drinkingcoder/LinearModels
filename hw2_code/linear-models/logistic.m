function w = logistic(X, y)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE

[P,N] = size(X);
X = [ones(1,N);X];
P = P+1;
y(y==-1) = 0;   %formula assume the other class labeled 0, but input label is -1

w = ones(P,1);

nRep = 1000;

lambda = 0.5;

threshold = 0.00000001;

D = eye(P);

p1 = 1./(1+exp(-w'*X));
grad_last = -sum(bsxfun(@times,X,y-p1),2);

for i=1:nRep
    direction = -D*grad_last;
    s = lambda*direction;
    w = w+s;
    
    p1 = 1./(1+exp(-w'*X)); %htheta, 1-by-N vector
    grad_cur = -sum(bsxfun(@times,X,y-p1),2);   %gradient, 1-by-N vector -> sum
    
    yk = grad_cur - grad_last;
    grad_last = grad_cur;
    
    D = D+(s*s')/(s'*yk)-(D*yk*yk'*D)/(yk'*D*yk);
    
 %   hes = bsxfun(@times,X,p1.*(1-p1))*X';

 %   w = w - lambda*hes\jac;
  %  w = w - lambda*(hes\jac);
    if(norm(grad_cur,2)<threshold) break;
end
end
