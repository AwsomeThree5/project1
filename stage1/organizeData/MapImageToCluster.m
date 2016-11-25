function [  ] = MapImageToCluster(  )
%MAPIMAGETOCLUSTER Summary of this function goes here
%   Detailed explanation goes here
vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
fishType = 'NoF';
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
dataFolder = [dataFolder, '\', fishType];
if ~exist(dataFolder, 'dir')
    disp('fish data path should look like : ');
    disp([blanks(3),'...vl_rootnn\data\fish\train\train']);
    return;
end

% include folders and subfolders of this project
currentPath = mfilename('fullpath');
gitPath = currentPath(1:strfind(currentPath, projectName)+numel(projectName)-1);
addpath(genpath(gitPath))

% find images names inside the data folder
allImages = dir([dataFolder,  '\*.jpg']);
allImages = {allImages(:).name}';

% find cluster folders
clusterNames = dir(dataFolder);
clusterNames(~[clusterNames.isdir]) = [];
clusterNames = {clusterNames(:).name}';
clusterNames(ismember(clusterNames, {'.', '..'})) = [];
%sort clusters by order
[~,ind] = sort(cellfun(@str2num, clusterNames));
clusterNames = clusterNames(ind);

%run through cluster folders, and generate a vector containing all
%clustered images names
clusteredImages.name =  cell(0,1);
clusteredImages.cluster = cell(0,1);
counter = 0;
numOfClusters = numel(clusterNames);
for clusterIdx = 1:numOfClusters
    disp(['cluster number : ', num2str(clusterIdx), '/', num2str(numOfClusters)]);
    clusterFolder = [dataFolder, '\', clusterNames{clusterIdx}];
    clusterImages = dir([clusterFolder, '\*.jpg']);
    clusterImages = {clusterImages.name}';
    counter = counter + numel(clusterImages);
    clusteredImages.name(end+1:end+numel(clusterImages)) = clusterImages;
    clusteredImages.cluster(end+1:end+numel(clusterImages)) = num2cell(clusterIdx * ones(size(clusterImages,1),1));
end

save(fullfile(fileparts(mfilename('fullpath')),'organizeData', ['clusteredImages_',fishType, '.mat']), 'clusteredImages');

end

