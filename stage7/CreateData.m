function [ ] = CreateData( )
%CREATEDATA Summary of this function goes here
%   Detailed explanation goes here
vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
desktopPath = 'C:\Users\David\Desktop';
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
data.camera = cell(0,1);
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
    data.camera(end+1:end+numel(clusteredImages.cluster)) = clusteredImages.cluster;
    data.name(end+1:end+numel(clusteredImages.name)) = clusteredImages.name;
    data.fishType(end+1:end+numel(fishTypeCellArray)) = fishTypeCellArray;
end

fishTypes = unique(data.fishType);

fishDataPath = fullfile(desktopPath,'YesNoFish','Fish');
excludedFishPath = fullfile(desktopPath,'YesNoFish','nof');
nofDataPath = fullfile(desktopPath,'YesNoFish','NoFish');

% run on fish data end extract the camera number, based on 'data' struct
dirStruct = dir([fishDataPath, '\*jpg']);
imageNames = {dirStruct.name}';
fishData.path = fullfile(fishDataPath, imageNames);
imageNames = cellfun(@(x) strsplit(x, {'_'}), imageNames(:), 'UniformOutput', false);
fishData.name = cellfun(@(x) strcat(x{1},'_', x{2}, '.jpg'), imageNames(:), 'UniformOutput', false);
fishIndices = cellfun(@(x) find(strcmpi(data.name, x)==1), fishData.name, 'UniformOutput', false);
fishData.camera = data.camera([fishIndices{:}]);
fishData.fishType = data.fishType([fishIndices{:}]);
fishData.label(1:numel(fishData.name)) = 1; % positive examples
% show the histogram if you want..
% figure;hist([fishData.camera{:}], 32)

% run on hand labeled data from originally 'fish' images, but finally
% labeld as 'no fish'
dirStruct = dir([excludedFishPath, '\*jpg']);
imageNames = {dirStruct.name}';
excludedFishData.path = fullfile(excludedFishPath, imageNames);
imageNames = cellfun(@(x) strsplit(x, {'_'}), imageNames(:), 'UniformOutput', false);
excludedFishData.name = cellfun(@(x) strcat(x{1},'_', x{2}, '.jpg'), imageNames(:), 'UniformOutput', false);
fishIndices = cellfun(@(x) find(strcmpi(data.name, x)==1), excludedFishData.name, 'UniformOutput', false);
excludedFishData.camera = data.camera([fishIndices{:}]);
excludedFishData.fishType = data.fishType([fishIndices{:}]);
excludedFishData.label(1:numel(excludedFishData.name)) = 0; % negative examples
% show the histogram if you want..
% hist([excludedFishData.camera{:}], 32)



% run on nof data end extract the camera number, based on 'data' struct. 
dirStruct = dir([nofDataPath, '\*jpg']);
imageNames = {dirStruct.name}';
noFishData.path = fullfile(nofDataPath, imageNames);
imageNames = cellfun(@(x) strsplit(x, {'_'}), imageNames(:), 'UniformOutput', false);
noFishData.name = cellfun(@(x) strcat(x{1},'_', x{2}, '.jpg'), imageNames(:), 'UniformOutput', false);
fishIndices = cellfun(@(x) find(strcmpi(data.name, x)==1), noFishData.name, 'UniformOutput', false);
noFishData.camera = data.camera([fishIndices{:}]);
noFishData.fishType = data.fishType([fishIndices{:}]);
noFishData.label(1:numel(noFishData.name)) = 0; % negative examples
% show the histogram if you want..
% figure;hist([noFishData.camera{:}], 32)


% now we start to build our new data struct. 
% first, we want to save only fishData & excludedFishData to the struct.
% they will be the train/val data.
% the noFishData will be used as a backup source for crops without any
% fish.
clear data
data.name = cat(1, fishData.name, excludedFishData.name);
data.camera = cat(1, fishData.camera, excludedFishData.camera);
data.fishType = cat(1, fishData.fishType, excludedFishData.fishType);
data.label = cat(2, fishData.label, excludedFishData.label);
data.path= cat(1, fishData.path, excludedFishData.path);

train = 0.8;
numOfCameras = max([data.camera{:}]);
% use a constant seed, to generate the same random numbers in every new run.
rng('default')
rng(1)
for cameraIdx = 1:numOfCameras
    disp (['camera index : ', num2str(cameraIdx)]);
    % once for the fish images
    fishIndices = find( [data.camera{:}] == cameraIdx & data.label == 1);
    randInd = randperm(numel(fishIndices));
    numOfTrain = floor(train * numel(fishIndices));
    numOfTest = numel(fishIndices) - numOfTrain;
    data.set(fishIndices(randInd(1:numOfTrain))) = ones(numOfTrain,1);
    data.set(fishIndices(randInd(numOfTrain+1:end))) = 2*ones(numOfTest,1);
    
    
    % and once for the noFish images
    noFishIndices = find( [data.camera{:}] == cameraIdx & data.label == 0);
    randInd = randperm(numel(noFishIndices));
    numOfTrain = floor(train * numel(noFishIndices));
    numOfTest = numel(noFishIndices) - numOfTrain;
    data.set(noFishIndices(randInd(1:numOfTrain))) = ones(numOfTrain,1);
    data.set(noFishIndices(randInd(numOfTrain+1:end))) = 2*ones(numOfTest,1);
end

% now we shall add the 'noFish' dataset
data.name = cat(1, data.name, noFishData.name);
data.camera = cat(1, data.camera, noFishData.camera);
data.fishType = cat(1, data.fishType, noFishData.fishType);
data.label = cat(2, data.label, noFishData.label);
data.path= cat(1, data.path, noFishData.path);
data.set  = cat(2, data.set, 3*ones(numel(noFishData.name),1)');

save(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage7', 'data.mat'), 'data', '-v7.3')

end



