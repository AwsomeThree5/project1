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
imdb = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'imdb.mat'));
valInd = find(imdb.images.set == 2);

res = vl_simplenn(net, imdb.images.data(:,:,:,valInd));
[certainty,  predictions] = max(res(end).x, [], 3);

predictions = squeeze(squeeze(squeeze(predictions)));
groundTruth = imdb.images.labels(valInd);

[confusionMat,order] = confusionmat(groundTruth,predictions);




end

