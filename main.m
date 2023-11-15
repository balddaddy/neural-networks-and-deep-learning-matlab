% copyright Chang 2023.11.13
% This program is the code for the online book "neural network and deep
% learning", which using neural network to recognize handwriting digitals
clear; close all; clc;

%% prepare data set
load('mnist.mat');
n = training.count;
perm = randperm(n, 20);
for id = 1:20
%     pid = perm(id);
    pid = id;
    subplot(4,5,id);
    imshow(training.images(:,:,pid));
    drawnow;
    title (sprintf ("The %d ", training.labels(pid)));
end


% pick 5000 of training images as training_data, 1000 as test_data
training_data.count = training.count;
training_data.width = training.width;
training_data.height = training.height;
training_data.images = reshape(training.images,training.width*training.height,...
    1,training.count);
training_data.labels = training.labels;
test_data.count = test.count;
test_data.width = test.width;
test_data.height = test.height;
test_data.images = reshape(test.images,test.width*test.height,...
    1, test.count);
test_data.labels = test.labels;

%% create the network
epochs = 30;
mini_batch_size = 10;
eta = 3.0;
net = Network([784,30,10]);
net.SGD(training_data, epochs, mini_batch_size, eta, test_data);
