function [ output_args ] = FindBoundingBoxCentroids( input_args )
% manually find the bounding box centroid point, running on images from
% each camera view

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

% load struct of images indicating the path and cluster.
data = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'));
data = data.data;


numOfCameras = max([data.label{:}]);
boundingBoxCentroid = zeros(numOfCameras, 2); % for (x,y)
for cameraIdx = 1:numOfCameras
    clusteredImagesInd = find([data.label{:}] == cameraIdx);
    figure;
    for imageIdx = 1:1:numel(clusteredImagesInd)
        imagePath = fullfile(dataFolder, data.fishType{clusteredImagesInd(imageIdx)}, data.name{clusteredImagesInd(imageIdx)});
        img = imread(imagePath);
        imshow(img)
        waitforbuttonpress() % look at some of the images...
    end
    % BREAKPOINT HERE
        boundingBoxCentroid
    
end

save boundingBoxCentroid boundingBoxCentroid;



end

