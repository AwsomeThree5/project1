function [  ] = CheckTrainingValidity(  )
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
testDataFolder = fullfile(vl_rootnnPath, 'data\fish\test_stg1\test_stg1');

if ~exist(testDataFolder, 'dir')
    disp('fish data path should look like : ');
    disp([blanks(3),'...vl_rootnn\data\fish\train\train']);
    return;
end

% include folders and subfolders of this project
currentPath = mfilename('fullpath');
gitPath = currentPath(1:strfind(currentPath, projectName)+numel(projectName)-1);
addpath(genpath(gitPath))


% lets start
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

camNet = load('camNet.mat');
camNet = camNet.net;
% get only the forward net
fishNet = load('fishNet.mat');
fishNet = fishNet.net;
% need to change the type from 'softmaxloss' to 'softmax'
fishNet.layers{end}.type = 'softmax';
camNet.layers{end}.type = 'softmax';

% read all images
allImages = dir([testDataFolder,  '\*.jpg']);
allImages = {allImages(:).name}';

% load mean data
camMean = load('camDataMean.mat');
camMean = camMean.dataMean;

% load mean data
fishMean = load('fishDataMean.mat');
fishMean = fishMean.dataMean;

% for resizing images before they go into the camClassifier
finalRatio = 0.57; % = mean(ratio)
w1 = 100;
h1 = floor(w1*finalRatio);
w2 = 400;
h2 = w2;

testSoftmax = cell(numel(allImages),1);
camPredictions = zeros(numel(allImages),1);

if ~exist('testSoftmax.mat', 'file')
    for imageIdx = 1:1:numel(allImages)
        disp(['image number : ', num2str(imageIdx), '/', num2str(numel(allImages))]);
        img = single(imread(allImages{imageIdx}));
        resizedImg = imresize(img, [h1,w1]);
%         resizedImg = resizedImg - camMean;
        res = vl_simplenn(camNet, resizedImg);
        [~,  prediction] = max(res(end).x, [], 3);
        camPredictions(imageIdx) = prediction; 
        cameraNumber = prediction;
        cameraCentroid = boundingBoxCentroid(cameraNumber, :);
        xMin = max(cameraCentroid(1)-w2/2, 1);
        xMax = min(cameraCentroid(1)+w2/2, size(img, 2));
        % make sure the cropped size is WxH
        if xMin == 1
            xMax = w2+1;
        elseif xMax == size(img,2)
            xMin = xMax - w2;
        end
        yMin = max(cameraCentroid(2)-h2/2, 1);
        yMax = min(cameraCentroid(2)+h2/2, size(img, 1));
        % make sure the cropped size is WxH
        if yMin == 1
            yMax = h2+1;
        elseif yMax == size(img,1)
            yMin = yMax - h2;
        end
%         img = single(imread(allImages{imageIdx}));
        croppedImg = img(yMin+1:yMax, xMin+1:xMax , :);
        croppedImg = croppedImg - fishMean;
%         imshow(croppedImg);
%         waitforbuttonpress();
        res1 = vl_simplenn(fishNet, croppedImg);
        testSoftmax{imageIdx} = res1(end).x;
    end
    save testSoftmax testSoftmax
    clear testSoftmax;
end
load(fullfile(fileparts(currentPath), 'testSoftmax.mat'));
testSoftmaxMatrix = squeeze([testSoftmax{:}]);
[scores, testPredictions] = max(testSoftmaxMatrix');

csvTable = readtable(fullfile(vl_rootnnPath, 'data\fish\sample_submission_stg1.csv\sample_submission_stg1.csv'));
for imageIdx = 1:numel(allImages)
    disp(['image number : ',num2str(imageIdx),'/', num2str(numel(allImages))]);
    if strcmp(csvTable{imageIdx, 1}, allImages{imageIdx})
        for index = 1:size(testSoftmaxMatrix,2)
            csvTable{imageIdx, index+1} = testSoftmaxMatrix(imageIdx, index);
        end
    else
        disp('DEBUG HEREERERERE')
    end
end
writetable(csvTable, fullfile(vl_rootnnPath, 'data\fish\sample_submission_stg1.csv\test_submission_stg1.csv'))
disp('end');




end