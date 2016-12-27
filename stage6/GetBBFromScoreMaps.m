function [ ] = GetBBFromScoreMaps( )
%GETBBFROMSCOREMAPS Summary of this function goes here
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
scoresFolder = fullfile(vl_rootnnPath, 'data\fish\train\scores');
cropsFolder = fullfile(vl_rootnnPath,'data\fish\train\YesNoFish');

if ~exist(dataFolder, 'dir')
    disp('fish data path should look like : ');
    disp([blanks(3),'...vl_rootnn\data\fish\train\train']);
    return;
end

% include folders and subfolders of this project
currentPath = mfilename('fullpath');
gitPath = currentPath(1:strfind(currentPath, projectName)+numel(projectName)-1);
addpath(genpath(gitPath))

% load the data
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'data.mat'));

% load the bounding boxes for each camera
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

% find all nets with scores folders
scoreNet = dir(scoresFolder);
scoreNet = {scoreNet.name}';
scoreNet(ismember(scoreNet, {'.','..'})) = [];


% APPLY SELECTIVE SEARCH
Data = [];
for imageIdx = 1:numel(data.name)
    if strcmpi(data.fishType{imageIdx}, 'Nof')
        disp(['image number : ',num2str(imageIdx), '/', num2str(numel(data.name))]);
        scoreMap= GetScoreImage(scoreNet,data,dataFolder,scoresFolder, imageIdx, boundingBoxCentroid);
        imageName = strsplit(data.name{imageIdx},'.');
        imageName = imageName{1};
        matchingImage = imread(fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx}));
        % get the relevant crop according to the camera label
        [xMin, xMax, yMin, yMax] = GetCameraCropIndices( matchingImage, boundingBoxCentroid, data.camera{imageIdx}, 500 ,500);

        boundingBoxes = RunSelectiveSearch(matchingImage,xMin,xMax,yMin,yMax );

    %     boundingBoxes = FilterBoxesUsingScoreMap(boundingBoxes, scoreMap, matchingImage);
        disp([blanks(2),'final : ',num2str(size(boundingBoxes,1)), ' boxes']);
        WriteBoxesToMemory(cropsFolder, boundingBoxes, matchingImage,data.fishType{imageIdx}, imageName);
        Data.fishType{imageIdx} = data.fishType{imageIdx};
        Data.name{imageIdx} =  data.fishType{imageIdx};
        Data.boundingBoxes{imageIdx} = boundingBoxes;
        Data.NumOfBoxes{imageIdx} = size(boundingBoxes,1);
    end
end
save(fullfile(cropsFolder,'DataNof.mat'), 'Data', '-v7.3');
end

function [] = WriteBoxesToMemory(cropsFolder, boxes, img,fishType, imageName)

if strcmpi(fishType, 'Nof')
    cropsFolder = fullfile(cropsFolder,'NoFish');
else
    cropsFolder = fullfile(cropsFolder,'Fish');
end
% save all crops with meaningfull names :)
for bboxIdx = 1:size(boxes,1)
    imageCrop = img(boxes(bboxIdx, 1):boxes(bboxIdx, 3), boxes(bboxIdx, 2):boxes(bboxIdx, 4),:);
    imwrite(imageCrop, fullfile(cropsFolder,[imageName,'_',num2str(bboxIdx), '.jpg']));
end

end
function [boxes] = FilterBoxesUsingScoreMap(boxes, scoreMap, img)

% run through all boxes and rank them according to the scoreMap in their
% region
boxesSumScore = zeros(size(boxes,1), 1);
scoreMapSum = sum(sum(scoreMap));
thresh = 0.5;
for boxIdx= 1:size(boxes,1)
    boxesSumScore(boxIdx) = sum(sum(scoreMap(boxes(boxIdx, 1):boxes(boxIdx, 3), ...
    boxes(boxIdx, 2):boxes(boxIdx, 4))));
end
goodBoxes = find(boxesSumScore/scoreMapSum > thresh);
% ShowRectsWithinImage(boxes(goodBoxes,:), 2, 2, img)
boxes = boxes(goodBoxes,:);
end
function [xMin,xMax,yMin,yMax] = GetCameraCropIndices( img, boundingBoxCentroid, cameraNumber, h, w)

cameraCentroid = boundingBoxCentroid(cameraNumber, :);
xMin = max(cameraCentroid(1)-w/2, 1);
xMax = min(cameraCentroid(1)+w/2, size(img, 2));
% make sure the cropped size is WxH
if xMin == 1
    xMax = w+1;
elseif xMax == size(img,2)
    xMin = xMax - w;
end
yMin = max(cameraCentroid(2)-h/2, 1);
yMax = min(cameraCentroid(2)+h/2, size(img, 1));
% make sure the cropped size is WxH
if yMin == 1
    yMax = h+1;
elseif yMax == size(img,1)
    yMin = yMax - h;
end

end


function [finalScoreMap] = GetScoreImage(scoreNet,data,dataFolder,scoresFolder, imageIdx, boundingBoxCentroid)
% calculate binary scoreImages
histCountsNoFish = zeros(256,numel(scoreNet));
histCountsFish = zeros(256,numel(scoreNet));

% for imageIdx = 1:100:numel(data.name)

% get image and bounding box according to the camera index
imageName = strsplit(data.name{imageIdx}, '.');
matchingImage = imread(fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx}));
[xMin, xMax, yMin, yMax] = GetCameraCropIndices( matchingImage, boundingBoxCentroid, data.camera{imageIdx}, 600 ,600);

% hold only values from scoreImages, in the area of the camera crop
croppedScoreMap = zeros(size(matchingImage,1), size(matchingImage,2), numel(scoreNet),'single');
for scoreNetIdx = 1:numel(scoreNet)
    
    % read the score map and rescale it to match the image
    scoreMap= imread(fullfile(scoresFolder,scoreNet{scoreNetIdx}, data.fishType{imageIdx}, [imageName{1},'.bmp']));
    enlargedScoreMap = imresize(scoreMap, [size(matchingImage,1), size(matchingImage,2)], 'bicubic');
    
    % compute the histogram
    [counts, ~] = imhist(enlargedScoreMap(yMin:yMax-1, xMin:xMax-1), 256);
    if ~strcmp(data.fishType{imageIdx} ,'NoF')
        histCountsFish(:,scoreNetIdx) = histCountsFish(:,scoreNetIdx) + counts;
    else
        histCountsNoFish(:,scoreNetIdx) = histCountsNoFish(:,scoreNetIdx) + counts;
    end
    
    % cropped score map according to the camera index
    currentScoreMap = enlargedScoreMap(yMin:yMax-1, xMin:xMax-1);
    
    % create a binary image
    threshHold = graythresh(currentScoreMap);
    currentScoreMap(currentScoreMap >= threshHold*255) = 255;
    currentScoreMap(currentScoreMap < threshHold*255) = 0;
    croppedScoreMap(yMin:yMax-1,xMin:xMax-1,scoreNetIdx) = currentScoreMap;
    
    %         imshow(croppedScoreMap(:,:,scoreNetIdx)/255);
    %         title(imageName{1});
    %         waitforbuttonpress();
    
end
finalScoreMap = mean(croppedScoreMap, 3);
%     imshow(matchingImage)
%     waitforbuttonpress();
%     imshow(finalScoreMap/255);
finalScoreMap(finalScoreMap <= 4*255/numel(scoreNet)) = 0;
finalScoreMap(finalScoreMap > 4*255/numel(scoreNet)) = 255;
%     imshow(finalScoreMap/255);
%     title(imageName{1});
%     waitforbuttonpress();

% end
% normalizedCountNof = histCountsNoFish / sum(histCountsNoFish, 1);
% normalizedCount = histCountsFish / sum(histCountsFish, 1);

end