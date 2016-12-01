function [ confusionMat ] = CalculateConfusionMatrix( netPath )
% insert the path to the net.mat file and get the confusion matrix

vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
projectName = 'project1';

addpath(genpath(vl_rootnnPath))
try
    warning off;
    vl_setupnn();
catch
    disp('you need to initialize the matconvnet path first');
    return;
end
dataFolder = fullfile(vl_rootnnPath, 'data\fish\train\train');

if ~exist(dataFolder, 'dir')
    disp('fish data path should look like : ');
    disp([blanks(3),'...vl_rootnn\data\fish\train\train']);
    return;
end

% include folders and subfolders of this project
currentPath = mfilename('fullpath');
gitPath = currentPath(1:strfind(currentPath, projectName)+numel(projectName)-1);
addpath(genpath(gitPath))


% lets start
% netPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage2\net-epoch-13.mat';
net = load('net.mat');
% get only the forward net
net = net.net;
% need to change the type from 'softmaxloss' to 'softmax'
net.layers{end}.type = 'softmax';
% load imdb
imdb = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'imdb.mat'));
valInd = find(imdb.images.set == 2);

allPredictions = zeros(numel(valInd), 1);
allCertainty = zeros(numel(valInd), 1);
batchSize = 1;
for batchIdx = 1:batchSize:numel(valInd)
    disp(['batch index : ',num2str(batchIdx)]);
    res = vl_simplenn(net, imdb.images.data(:,:,:,valInd(batchIdx)));
    [certainty,  predictions] = max(res(end).x, [], 3);
    allCertainty(batchIdx) = certainty;
    allPredictions(batchIdx) = predictions;
end
allPredictions = squeeze(squeeze(squeeze(allPredictions)));
allPredictions = allPredictions';
groundTruth = imdb.images.labels(valInd);

[confusionMat,order] = confusionmat(groundTruth,allPredictions);




end

