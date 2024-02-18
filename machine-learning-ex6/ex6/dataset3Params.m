function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

predict_C = [0.01 0.05 0.1 0.5 1 5 10 50];
predict_sigma = [0.01 0.05 0.1 0.5 1 5 10 50];

predict_error = 100;

for i = 1 : length(predict_C),
	current_C = predict_C(i);
	for j = 1: length(predict_sigma),
		current_sigma = predict_sigma(j);
		model = svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
		predictions = svmPredict(model, Xval);
		current_error = mean(double(predictions ~= yval));

		if current_error < predict_error,
			predict_error = current_error;
			C = current_C;
			sigma = current_sigma;
		end
	end
end






% =========================================================================

end
