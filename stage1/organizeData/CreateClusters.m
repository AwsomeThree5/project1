function [ output_args ] = CreateClusters( fishType )
% inputs: fishClass
% output: creates folders for each cluster, with the corresponding images

vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
projectName = 'project1';

addpath(genpath(vl_rootnnPath))
try
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

% try reading the mat file (if it exists)
try
    load(fullfile(fileparts(currentPath), ['clusteredImages_',fishType, '.mat']));
catch ex
    disp(ex.message);
end

numClusters = max([clusteredImages.cluster{:}]);
% create folders for all clusters
for clusterIdx = 1:numClusters
    clusterPath = [dataFolder, '\', num2str(clusterIdx)];
    if ~exist(clusterPath,'dir')
        mkdir(clusterPath);
    else
        disp('directory exists, what do you wish to do ???');
        waitforbuttonpress();
        % option to remove the folder and it's containings
        rmdir(clusterPath,'s');
    end
end


% copy images to the cluster folders
numOfImages = numel(clusteredImages.name);
for imageIdx = 1:numOfImages
    disp(['writing image number : ', num2str(imageIdx), '/', num2str(numOfImages)]);
    clusterPath = [dataFolder, '\', num2str(clusteredImages.cluster{imageIdx})];
    image = imread([dataFolder, '\', clusteredImages.name{imageIdx}]);
    imwrite(image, [clusterPath, '\',  clusteredImages.name{imageIdx}]);
end




end

