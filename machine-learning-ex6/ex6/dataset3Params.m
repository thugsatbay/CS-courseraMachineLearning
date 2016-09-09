function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
testC=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
testSigma=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
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
lenC=size(testC,1);
lenSigma=size(testSigma,1);
errorArray=zeros(lenC,lenSigma);
for i=1:lenC
    for j=1:lenSigma
        fprintf('validate iteraton C=%f Sigma=%f',i,j);
        model= svmTrain(X, y, testC(i), @(x1, x2) gaussianKernel(x1, x2, testSigma(j)));
        predictions = svmPredict(model, Xval);
        errorArray(i,j)=mean(double(predictions ~= yval));
    end
end
[a,indr]=min(errorArray);
[b,indc]=min(a);
C=testC(indr(indc));
sigma=testSigma(indc);
%C                              %final C value
%sigma                          %final sigma value
%errorArray
%errorArray(indr(indc),indc)  %lowest error cost value

% =========================================================================

end
