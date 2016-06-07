function [w, iter] = perceptron(X, y)
%PERCEPTRON Perceptron Learning Algorithm.
%
%   INPUT:  X: training sample features, P-by-N matrix.
%           y: training sample labels, 1-by-N row vector.
%
%   OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
%           iter: number of iterations
%

% YOUR CODE HERE

X = [ones(1,size(X,2));X];  %Notice here!!!!
X = bsxfun(@times,X,y);     % xi = xi*yi
w = ones(size(X,1),1);
lambda = 1;
max_err = size(X,2);

iter = 0;
for i=1:1000
    loss = w'*X;
    loss(loss>0) = 0;
    loss_total = sum(loss<0);
    if(loss_total == 0) 
         return;
    end
    if(loss_total < max_err)
        max_err = loss_total;
        w_sel = w;
    end
    iter = iter+1;
    
    error = sign(loss);
    grad_w = sum(bsxfun(@times,error,X),2);
    w = w - lambda*grad_w;
    
  %  if(loss_last <= loss_total*0.8)
  %      lambda = lambda*0.5;
  %  elseif(loss_last >= loss_total*1.25)
  %      lambda = lambda*2;
  %  end
   % plotdata(xx, y,w_f,w, 'Pecertron');
   % pause();
end
    w = w_sel;