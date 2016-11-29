function [ ] = CreateData( )
%CREATEDATA Summary of this function goes here
%   Detailed explanation goes here
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

% read all matFiles, created in 'stage1', holding the images per class and
% their cluster.
stage1Path = [gitPath, '\stage1\organizeData'];
matFileDir = dir([stage1Path, '\*.mat']);
matFileNames = {matFileDir(:).name}';

data.name = cell(0,1);
data.label = cell(0,1);
data.fishType = cell(0,1);
for matFileIdx = 1:numel(matFileNames)
    % load mat file for images in a fish folder
    clusteredImages = load([stage1Path, '\', matFileNames{matFileIdx}]);
    clusteredImages = clusteredImages.clusteredImages;
    % find fishType name
    fishType = strsplit(matFileNames{matFileIdx}, {'_', '.'});
    fishType = fishType{2};
    clear fishTypeCellArray;
    % create a cell array of the same string
    [fishTypeCellArray{1:numel(clusteredImages.cluster)}] = deal(fishType);
    fishTypeCellArray = fishTypeCellArray';
    % save the data
    data.label(end+1:end+numel(clusteredImages.cluster)) = clusteredImages.cluster;
    data.name(end+1:end+numel(clusteredImages.name)) = clusteredImages.name;
    data.fishType(end+1:end+numel(fishTypeCellArray)) = fishTypeCellArray;
end

% run on clusters and seperate images (randomly) to train and test 
train = 0.8;
numOfClusters = max([data.label{:}]);
numOfExamples = numel(data.label);

% use a constant seed, to generate the same random numbers in every new run.
rng('default')
rng(1)
data.set = cell(numOfExamples,1);
for clusterIdx = 1:numOfClusters
    indices = find([data.label{:}] == clusterIdx);
    randInd = randperm(numel(indices));
    numOfTrain = floor(train * numel(indices));
    numOfTest = numel(indices) - numOfTrain;
    data.set(indices(randInd(1:numOfTrain))) = num2cell(ones(numOfTrain,1));
    data.set(indices(randInd(numOfTrain+1:end))) = num2cell(2*ones(numOfTest,1));
end

% reorder data to train and then val data

trainInd = find([data.set{:}] == 1);
testInd = find([data.set{:}] == 2);
indices = cat(2, trainInd, testInd);
data.fishType = data.fishType(indices);
data.name = data.name(indices);
data.label = data.label(indices);
data.set= data.set(indices);
save(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'), 'data', '-v7.3')



end

