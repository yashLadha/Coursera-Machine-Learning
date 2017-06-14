function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[J, grad] = costFunction(theta, X, y);

temp_j = lambda*sum(theta(2:end).^2);
temp_j = temp_j/(2*m);

J = J + temp_j;

temp_g = zeros(size(theta));
for i = 2:size(theta)
   temp_g(i) = (lambda/m)*(theta(i));
endfor
grad = grad + temp_g;


% =============================================================

end
