function dSigma = sigmoid_prime(z)
% Derivative of the sigmoid function
dSigma = sigmoid(z).*(ones(size(z,1),size(z,2)) - sigmoid(z));
end