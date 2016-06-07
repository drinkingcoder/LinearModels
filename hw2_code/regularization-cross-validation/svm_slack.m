function [w,num] = svm_slack(X,y)

[P,N] = size(X);
C = 1;

X = [ones(1,N);X];
H = [eye(P+1,P+1),zeros(P+1,P);zeros(P,2*P+1)];
f = [zeros(P+1,1);C*ones(P,1)];
A = [-bsxfun(@times,X,y)',-eye(P);zeros(P,P+1),-eye(P)];
b = [-ones(N,1);zeros(P,1)];
w = quadprog(H,f,A,b);
w = w(1:P+1);
num = sum(abs(1-abs(w'*X))<=0.0001);
end