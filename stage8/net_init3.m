function [ net ] = net_init3( varargin )
%NET_INIT3 Summary of this function goes here
%   Detailed explanation goes here


vggNet = load(fullfile(vl_rootnn, 'imagenet-vgg-verydeep-16'));
h = varargin{1};
w = varargin{2};
c = varargin{3};

opts.networkType = 'simplenn' ;


% same seed always
s = RandStream('mt19937ar','Seed',0);
lr = [1 2] ;
HH = [14,1];
WW = [14,1];
C = [512,4096];
F = [4096,2];
weightsScale = 2./sqrt(HH.*WW.*(F+C));
% Define network CIFAR10-quick
net.layers = {} ;
net.layers = vggNet.layers(1:24);
for layerIdx = 1:numel(net.layers)
    if strcmp(net.layers{layerIdx}.type, 'conv')
        net.layers{layerIdx}.learningRate = [0 0];
    end
end

% Block 1


% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'max', ...
%                            'pool', [3 3], ...
%                            'stride', 2, ...
%                            'pad', [0 1 0 1]) ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{weightsScale(1)*randn(HH(1),WW(1),C(1),F(1), 'single'), zeros(1,F(1),'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {{ones(1,1,C(2), 'single'), zeros(1,C(2),'single')}});
net.layers{end+1} = struct('type', 'relu');%, 'leak', 0.02) ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{weightsScale(2)*randn(HH(2),WW(2),C(2),F(2), 'single'), zeros(1,F(2),'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;


% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [h w c] ;
net.meta.trainOpts.learningRate = 0.002*[10*ones(1,1), 0.3*ones(1,2) 0.1*ones(1,2) 0.01*ones(1,3), 0.005*ones(1,4)] ;
net.meta.trainOpts.weightDecay = 0.00001 ;
net.meta.trainOpts.batchSize = 100;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'error') ;
    otherwise
        assert(false) ;
end


end

