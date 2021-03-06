function [ net ] = fish_init( varargin )
%CNN_CAR_INIT_IMAGENET Summary of this function goes here
%   Detailed explanation goes here



h = varargin{1};
w = varargin{2};
c = varargin{3};

opts.networkType = 'simplenn' ;
% opts = vl_argparse(opts, varargin) ;


% same seed always
s = RandStream('mt19937ar','Seed',0);
lr = [.1 2] ;
HH = [7,7,7,14];
WW = [7,7,7,25];
C = [3,32,64,64];
F = [32,64,64,32];
weightsScale = 2./sqrt(HH.*WW.*(F+C));
% Define network CIFAR10-quick
net.layers = {} ;

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
    'pad', 3) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{weightsScale(2)*randn(HH(2),WW(2),C(2),F(2), 'single'), zeros(1,F(2),'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 3) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{weightsScale(3)*randn(HH(3),WW(3),C(3),F(3), 'single'), zeros(1,F(3),'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 3) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {{ones(1,1,C(4), 'single'), zeros(1,C(4),'single')}});
net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'dropout',...
%                             'rate', 0.5);


% Block 5
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{weightsScale(4)*randn(HH(4),WW(4),C(4),F(4), 'single'), zeros(1,F(4),'single')}}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [h w c] ;
net.meta.trainOpts.learningRate = 0.2*[0.05*ones(1,2) 0.01*ones(1,28) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
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



