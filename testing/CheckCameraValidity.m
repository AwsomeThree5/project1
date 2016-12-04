function [  ] = CheckCameraValidity(  )
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
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

camNet = load('camNet.mat');
camNet = camNet.net;
% need to change the type from 'softmaxloss' to 'softmax'
camNet.layers{end}.type = 'softmax';

% % read all images
% allImages = dir([dataFolder,  '\*.jpg']);
% allImages = {allImages(:).name}';


% for resizing images before they go into the camClassifier
finalRatio = 0.57; % = mean(ratio)
w1 = 100;
h1 = floor(w1*finalRatio);

load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'));
N = numel(data.label);
if 1
    for imageIdx = 1:N
        if round(imageIdx/100) == imageIdx/100
            disp(['image number : ',num2str(imageIdx)]);
        end
        imagePaths{imageIdx} = fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx});    
    end
end

camPredictions = zeros(numel(imagePaths),1);
outData = h5read(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'outData.h5'), ['/',fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2')]);
if ~exist('camPredictions.mat', 'file')
    for imageIdx = 1:1:numel(imagePaths)
        disp(['image number : ', num2str(imageIdx)]);
        img = single(imread(imagePaths{imageIdx}));
        resizedImg = imresize(img, [h1,w1]);
%         resizedImg = resizedImg - camMean;
        res = vl_simplenn(camNet, resizedImg);
        [~,  prediction] = max(res(end).x, [], 3);
        camPredictions(imageIdx) = prediction; 

        res2 = vl_simplenn(camNet, outData(:,:,:,imageIdx));
        [~,  prediction2] = max(res2(end).x, [], 3);
        camPredictions2(imageIdx) = prediction2;
    end
    save camPredictions camPredictions
    save camPredictions2 camPredictions2
end
load(fullfile(fileparts(currentPath), 'camPredictions.mat'));
load(fullfile(fileparts(currentPath), 'camPredictions2.mat'));

disp(['camera success : ', num2str(numel(find(camPredictions == [data.label{:}]'))/numel(data.label))]);
disp(['camera success from imdb: ', num2str(numel(find(camPredictions2' == [data.label{:}]'))/numel(data.label))]);

% csvTable = readtable(fullfile(vl_rootnnPath, 'data\fish\sample_submission_stg1.csv\sample_submission_stg1.csv'));
% for imageIdx = 1:numel(allImages)
%     if strcmp(csvTable{imageIdx, 1}, allImages{imageIdx})
%         for index = 1:size(softmaxMatrix,2)
%             csvTable{imageIdx, index+1} = softmaxMatrix(imageIdx, index);
%         end
%     else
%         disp('DEBUG HEREERERERE')
%     end
% end
% writetable(csvTable, fullfile(vl_rootnnPath, 'data\fish\sample_submission_stg1.csv\test_submission_stg1.csv'))
% disp('end');




end