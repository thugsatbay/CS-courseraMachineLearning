function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
% temp=zeros(2, 1);
%     temp(1)=theta(1)-alpha*(1/m)*sum(y-(X*theta));
% temp(2)=theta(2)-alpha*(1/m)*sum((y-(X*theta)).*X(:,2));
% theta=temp;

 % In every iteration calculate hypothesis
   hypothesis=theta(2).*X(:,2)+theta(1);

   % Update theta variables
   temp2=theta(2) - alpha * (1/m)* sum((hypothesis-y).* X(:,2));
   temp1=theta(1) - alpha * (1/m) *sum(hypothesis-y);

   theta(1)=temp1;
   theta(2)=temp2;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('%f %f %f \n', theta(1), theta(2), J_history(iter));

end

end
