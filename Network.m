% Network class
classdef Network < handle % 必须继承自handle类，其properties值才会更新
    properties
        num_layers = 0;
        sizes = 0;
        biases = [];
        weights = [];
    end
    methods
        %% Initialize network
        function obj = Network(sizes)
            if nargin == 1
                obj.num_layers = numel(sizes);
                obj.sizes = sizes;
                for lid = 1:obj.num_layers - 1
                    obj.biases{lid} = randn(obj.sizes(lid+1),1);
                    obj.weights{lid} = randn(obj.sizes(lid+1),sizes(lid));
                end
            else
                disp("please input one array at a time!");
            end
        end
        %% return the output of the network with input x
        function y = feedforward(obj, x)
            y = x;
            for lid = 1:obj.num_layers - 1
                y = sigmoid(obj.weights{lid} * y + obj.biases{lid});
            end
        end
        %% stochastic gradient descent
        % training the neural network using mini-batch stochastic gradient
        % descent. The 'training_data' is a list of pairs of matrices contianing
        % training input and desired output. The other non-optional
        % parameters are self-explanatory. If 'test_data' is not empty, the
        % network will be evaluated against the test data after each epoch,
        % and partial progress print out. This is useful for tracking
        % progreass, but slow things down substatnially
        function SGD(obj, training_data, epochs, mini_batch_size, eta, test_data)
            n = training_data.count;
            for eid = 1:epochs
                shuffle_id = randperm(n, n);
                for did = 1 : floor(n/mini_batch_size)
                    mini_batch_id = shuffle_id((did-1)*mini_batch_size+1:did*mini_batch_size);
                    mini_batch.count = mini_batch_size;
                    mini_batch.images = training_data.images(:,:,mini_batch_id);
                    mini_batch.labels = training_data.labels(mini_batch_id,1);
                    %                     figure(2); hold on;
                    %                     for id = 1:mini_batch_size
                    %                         %     pid = perm(id);
                    %                         subplot(2,5,id);
                    %                         images = mini_batch.images(:,:,id);
                    %                         images = reshape(images, 28, 28, 1);
                    %                         imshow(images);
                    %                         drawnow;
                    %                         title (sprintf ("The %d ", mini_batch.labels(id)));
                    %                     end
                    obj.updata_mini_batch(mini_batch, eta);
                end
                if nargin > 5
                    fprintf("Epoch %d: %d / %d\n", eid, obj.evaluate(test_data), ...
                        test_data.count);
                else
                    fprint("Epoch %d complete\n", eid);
                end
            end
        end
        function updata_mini_batch(obj, mini_batch, eta)
            % update the network's weights and biases by apllying gradient descent
            % using backpropogating to a single mini batch. The 'mini_batch' is a struct
            % including images and labels. The 'eta' is the learning rate));
            for lid = 1:obj.num_layers - 1
                nabla_b{lid} = zeros(numel(obj.biases{lid}), 1);
                nabla_w{lid} = zeros(size(obj.weights{lid},1), size(obj.weights{lid},2));
            end
            for id = 1:mini_batch.count
                [delta_nabla_b, delta_nabla_w] = obj.backprop(mini_batch.images(:,:,id), mini_batch.labels(id));
                for lid = 1:obj.num_layers - 1
                    nabla_b{lid} = nabla_b{lid} + delta_nabla_b{lid};
                    nabla_w{lid} = nabla_w{lid} + delta_nabla_w{lid};
                end
            end
            for lid = 1:obj.num_layers - 1
                obj.weights{lid} = obj.weights{lid} - eta/mini_batch.count .* nabla_w{lid};
                obj.biases{lid} = obj.biases{lid} - eta/mini_batch.count .* nabla_b{lid};
            end
        end
        function [nabla_b, nabla_w] = backprop(obj, x, y)
            % return "(nabla_b, nabla_w)" representing the gradient for the
            % cost function C_x.
            nabla_b = []; nabla_w = [];
            %% feedforward
            x = reshape(x, 784, 1);
            y = vectorized_result(y);
            activation = x;
            activations{1} = activation;
            zs = []; % to store all the z vectors
            for id = 2:obj.num_layers
                b = obj.biases{id-1};
                w = obj.weights{id-1};
                z = w * activation + b;
                zs{id - 1} = z;
                activation = sigmoid(z);
                activations{id} = activation;
            end
            %% backward pass
            delta = obj.cost_derivative(activations{end}, y) .* sigmoid_prime(zs{end});
            nabla_b{obj.num_layers - 1} = delta;
            nabla_w{obj.num_layers - 1} = delta * activations{end-1}.';
            %% Note that variable l in the loop below is used a little
            % differently to the notation in Chapter 2 of the book. Here,
            % l=1 means the last layer of neurons, l=2 is the second-last
            % layer, and so on. It's a renumbering of the scheme in the
            % book
            for lid = obj.num_layers - 1 : -1 : 2
                z = zs{lid - 1};
                sp = sigmoid_prime(z);
                delta = (obj.weights{lid}.' * delta) .* sp;
                nabla_b{lid - 1} = delta;
                nabla_w{lid - 1} = delta * activations{lid - 1}.';
            end
        end
        function rate = evaluate(obj, test_data)
            % return the number of test inputs for which the neural network
            % outputs the correct result. Note that the nueral network's
            % output is assumed to be the index of whichever neuron in the
            % final layer has the highest activation
            rate = 0;
            for id = 1:test_data.count
                output = obj.feedforward(test_data.images(:,:,id));
                [val,output] = max(output);
                output = output - 1;
                if output == test_data.labels(id)
                    rate = rate + 1;
                end
            end
        end
        function d = cost_derivative(obj, output_activations, y)
            % return the vector of partial derivatives $\partial C_x/\partial a$
            % for the output activations
            d = output_activations - y;
        end
    end
end