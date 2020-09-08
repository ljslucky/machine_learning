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
Theta1_grad = zeros(size(Theta1)); %26*401
Theta2_grad = zeros(size(Theta2)); %10*26

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
X = [ones(m,1) X];%5000*401

z2 = X * Theta1';%5000*401 401*25  = 5000*25
a2 = sigmoid(z2); %5000*25

a2 = [ones(size(a2,1),1) a2];%5000*26

z3 = a2 * Theta2';%5000*26  26*10 = 5000*10
hx = sigmoid(z3);% a3 5000*10

reg = lambda/(2*m)*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );
%hx(:,count) 5000*1 
%y1 5000*1
for count = 1:num_labels
  y1 = (y == count);
  J += -1/m*(y1'*log(hx(:,count)) + (1-y1)'*log(1-hx(:,count)));
endfor

J += reg;
D1 = 0;
D2 = 0;
 
for count = 1:m

  %step 1 £º 
  %Set the input layer¡¯s values (a(1)) to the t-th training example x(t). 
  %Perform a feedforward pass (Figure 2), 
  %computing the activations (z(2),a(2),z(3),a(3)) for layers 2 and 3. 
  %Note that you need to add a +1 term to ensure that the vectors of activations 
  %for layers a(1) and a(2) also include the bias unit
  a1 = X(count,:); %a1 1*401
  %a1 = [1 a1]; %a1 1*401
  z2 = Theta1 * a1';  % 25*401 401*1 = 25 * 1 
  a2 = sigmoid(z2); %25*1
  a2 = [1;a2]; %26*1 
  z3 = Theta2 * a2; %10*26 26*1  =10*1 
  a3 = sigmoid(z3); % 10*1
  
  %step 2
  yth = zeros(size(a3,1),1); % 10*1
  yth(y(count)) = 1; %10*1
  delta3 = a3 - yth; %10*1
   
  %step3 i do not know why? 
  delta2 = Theta2'*delta3.*a2.*(1 - a2);%26*10 * 10*1 .* 26*1 .* 26*1 = 26*1
  
  %step4 
  delta2 = delta2(2:end); % 25*1
 
  D1 = D1 + delta2 * a1;%25*401 + (25*1 * 1*401 ) = 25*401
  D2 = D2 + delta3 *(a2)';%10*26 + (10*1 * 1*26) = 10*26
  
endfor

%step5 
reg_theta1 = lambda/m*Theta1; % 25*401
reg_theta2 = lambda/m*Theta2; % 10*26
reg_theta1(:,1) = 0;
reg_theta2(:,1) = 0;


Theta1_grad = D1/m + reg_theta1;%25*401 + 25*401
Theta2_grad = D2/m + reg_theta2;%10*26 + 10*26


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
