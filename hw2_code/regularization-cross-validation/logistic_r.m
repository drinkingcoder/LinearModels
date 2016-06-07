function w = logistic_r(X, y, lambda)
%LR Logistic Regression.
%
%   INPUT:  X:   training sample features, P-by-N matrix.
%           y:   training sample labels, 1-by-N row vector.
%           lambda: regularization parameter.
%
%   OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
%

% YOUR CODE HERE
[P,N] = size(X);
X = [ones(1,N);X];
P = P+1;
y(y==-1) = 0;   %formula assume the other class labeled 0, but input label is -1

w = rand(P,1);

nRep = 1000;

rate = 0.01;

threshold = 0.00000001;
D = eye(P);

p1 = 1./(1+exp(-w'*X));
grad_last = -sum(bsxfun(@times,X,y-p1),2)+lambda*w;

for i=1:nRep
    p1 = 1./(1+exp(-w'*X)); %htheta, 1-by-N vector
    grad_cur = -sum(bsxfun(@times,X,y-p1),2)+lambda*w;
    w = w - rate*grad_cur;
    %{
    p1 = 1./(1+exp(-w'*X)); %htheta, 1-by-N vector
    jac = -sum(bsxfun(@times,X,y-p1),2)+lambda*w;   %gradient, 1-by-N vector -> sum
  %  disp(norm(jac,2));
    hes = bsxfun(@times,X,p1.*(1-p1))*X'+lambda*eye(P);

 %   w = w - lambda*hes\jac;
    if(lambda == 0)
        w = w - rate*jac;
    else
        w = w - rate*(hes\jac);
    end
   % if(norm(jac,2)<threshold) break;
    %}
    %{
    direction = -D*grad_last;
    s = rate*direction;
    w = w+s;
    
    p1 = 1./(1+exp(-w'*X)); %htheta, 1-by-N vector
    grad_cur = -sum(bsxfun(@times,X,y-p1),2)+lambda*w;   %gradient, 1-by-N vector -> sum
    
    yk = grad_cur - grad_last;
    grad_last = grad_cur;
    
    D = D+(s*s')/(s'*yk)-(D*yk*yk'*D)/(yk'*D*yk);
    %}
    norm(grad_cur,2)
    if(norm(grad_cur,2)<threshold) break;

end
end
