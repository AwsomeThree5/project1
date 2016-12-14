function [ ] = RunSelectiveSearch( )
%RUNSELECTIVESEARCH Summary of this function goes here
%   Detailed explanation goes here
% This demo shows how to use the software described in our IJCV paper:
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%

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

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation.
minSize = k;
sigma = 0.8;


% add params to the 'data' variable
% GatherHeadAndTailAnnotation();
% load struct of images indicating the path and cluster.
data = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'));
data = data.data;

numOfCameras = max([data.label{:}]);
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

h = 400;
w = h;
counter = 0;
for cameraIdx = 13:15%numOfCameras
    disp(['camera number : ',num2str(cameraIdx), '/', num2str(numOfCameras)]);
    clusteredImagesInd = find([data.label{:}] == cameraIdx);
    for imageIdx = 1:numel(clusteredImagesInd)
        disp([blanks(3), 'fish number : ',num2str(imageIdx), '/', num2str(numel(clusteredImagesInd))]);
        data.goodBoxes{clusteredImagesInd(imageIdx)} = [];
        data.badBoxes{clusteredImagesInd(imageIdx)} = [];
        if isempty(data.bbAnnotation{imageIdx})
            continue;
        end
        imagePath = fullfile(dataFolder, data.fishType{clusteredImagesInd(imageIdx)}, data.name{clusteredImagesInd(imageIdx)});
        img = imread(imagePath);
        cameraCentroid = boundingBoxCentroid(cameraIdx, :);
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
        centroidBox = [yMin, xMin, h, w];
        % Perform Selective Search
        [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(img, sigma, k, minSize, colorType, simFunctionHandles);
        boxes = BoxRemoveDuplicates(boxes);
        boxes = FilterIrrelevantBoxes(boxes, centroidBox, h, w, img);
        % Show boxes
%         ShowRectsWithinImage(boxes, 3, 3, img);
%         title(data.fishType{imageIdx});
        if ~isempty(data.bbAnnotation{imageIdx})
            goodBoxes = [];
            badBoxes = [];
            tempBoxes = boxes;
            tempBoxes(:,3) = tempBoxes(:,3) - tempBoxes(:,1)+1;
            tempBoxes(:,4) = tempBoxes(:,4) - tempBoxes(:,2)+1;
            for bbInd = 1:size(data.bbAnnotation{imageIdx}, 1)
                [value, ind] = sort(bboxOverlapRatio(tempBoxes, data.bbAnnotation{imageIdx}(bbInd, :), 'Min'), 'descend');
                goodBoxes = cat(1, goodBoxes, boxes(ind(value > 0.4), :));
                badBoxes = cat(1,badBoxes, boxes(ind(value <= 0.4), :));
%                 figure;imshow(img(data.bbAnnotation{imageIdx}(bbInd,1):data.bbAnnotation{imageIdx}(bbInd,1)+data.bbAnnotation{imageIdx}(bbInd,3),...
%                               data.bbAnnotation{imageIdx}(bbInd,2):data.bbAnnotation{imageIdx}(bbInd,2)+data.bbAnnotation{imageIdx}(bbInd,4), :));
            end
            % sort out duplicate boxes
            [~, ind] = unique(goodBoxes, 'rows');
            goodBoxes = goodBoxes(ind, :);
            
            badBoxes = boxes;
            isBadBox = ones(size(badBoxes,1), 1);
            for boxIdx = 1:size(badBoxes,1)
                for goodBoxIdx = 1:size(goodBoxes, 1)
                    if isequal(badBoxes(boxIdx, :), goodBoxes(goodBoxIdx, :))
                        isBadBox(boxIdx) = 0;
                    end
                end
            end
            badBoxes = badBoxes(logical(isBadBox), :);
%             ShowRectsWithinImage(newBoxes, 3, 3, img)
            if (size(goodBoxes,1) + size(badBoxes,1) ~= size(boxes,1))
                disp('WRONGGGG');
            end
            data.goodBoxes{clusteredImagesInd(imageIdx)} = goodBoxes;
            data.badBoxes{clusteredImagesInd(imageIdx)} = badBoxes;
            if (~isempty(goodBoxes))
                counter = counter + 1;
                
            end
        end
        close all;
        %
        % % Show blobs which result from first similarity function
        % hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1});
        % ShowBlobs(hBlobs, 5, 5, im);
    end
    % BREAKPOINT HERE
    %     boundingBoxCentroid
    
end

save(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data1.mat'), 'data');
disp(['counter : ', num2str(counter)]);






end

