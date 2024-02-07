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

my_log(:,1) = log(sigmoid(X*theta));
my_log(:,2) = log(1-(sigmoid(X*theta)));
my_y(:,1) = y;
my_y(:,2) = 1-y;
cost = my_log.*my_y;
J = (1/m)*(-sum(cost(:,1)) - sum(cost(:,2)))+...
	(lambda/(2*m))*sum(theta(2:end,1).^2);


grad(1,1) = (1/m)*sum((sigmoid(X*theta)-y).*X(:,1));
grad(2:end,1) = ((1/m)*((sigmoid(X*theta)-y)'*X(:,2:end)))'+(lambda/m)*theta(2:end);

grad = grad(:);


% =============================================================

end
