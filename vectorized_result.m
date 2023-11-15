function y = vectorized_result(x)
% transform the input number x (0~9) into a 10x1 vectors
y = zeros(10,1);
y(x+1) = 1;
end