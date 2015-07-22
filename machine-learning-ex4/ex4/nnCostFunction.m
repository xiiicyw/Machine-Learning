function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 Feedforward and cost function

% Theta1 = [25 401]
% Theta2 = [10 26]

X = [ones(m, 1) X];
a1 = X;
% a1 = [5000 401]
z2 = a1 * Theta1';
% z2 = [5000 25]
a2 = sigmoid(z2);
% a2 = [5000 25]
a2 = [ones(m, 1) a2];
% a2 = [5000 26]
z3 = a2 * Theta2';
% z3 = [5000 10]
a3 = sigmoid(z3);
% a3 = [5000 10]
h = a3;

% X = [5000 400]
% y = [5000 1]
% h = [5000 10]

Y = zeros(m, num_labels);  
% Y = [5000, 10]
for i = 1:m,  
    Y(i, y(i)) = 1;  
end;  

J = (1 / m) * ( -Y .* log(h) - (1-Y) .* log(1-h));
J = sum(sum(J));

% Part 1.4 Regulization

% Theta1 = [25 401]
% Theta2 = [10 26]

AllTheta1 = sum(sum(Theta1(:, 2:end).^2));
AllTheta2 = sum(sum(Theta2(:, 2:end).^2));
Reg = (lambda / (2 * m)) * (AllTheta1 + AllTheta2);
Reg = sum(sum(Reg));
J = J + Reg;


% Part 2.3 Backpropagation
Delta3 = a3 - Y;
% Delta3 = [5000 10]
Delta2 = Delta3 * Theta2(:,2:end) .* sigmoidGradient(z2);
% sigmoidGradient(z2) = [5000 25]
% Delta2 = [5000 25]

% a1 = [5000 401]
% a2 = [5000 26]
% a3 = [5000 10]

AllDelta1 = 0;
AllDelta2 = 0;
AllDelta1 = AllDelta1 + Delta2'*a1;
AllDelta2 = AllDelta2 + Delta3'*a2;
% AllDelta1 = [25 401]
% AllDelta2 = [10 26]

Theta1_grad = (1 / m) * AllDelta1;
Theta2_grad = (1 / m) * AllDelta2;
% Theta1_grad = [25 401]
% Theta1_grad = [10 26]

% Part 2.5 Regularized Neural Network
RegTheta1 = (lambda / m) * Theta1;
RegTheta2 = (lambda / m) * Theta2;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + RegTheta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + RegTheta2(:, 2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
