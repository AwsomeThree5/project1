function [ confusionMat ] = CalculateConfusionMatrix( )
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
net = load('fishNet.mat');
% get only the forward net
net = net.net;
% need to change the type from 'softmaxloss' to 'softmax'
net.layers{end}.type = 'softmax';
% load imdb
outData = h5read(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'outData.h5'), ['/', fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5')]);
imdb = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'imdb.mat'));
imdb = imdb.imdb;
valInd = find(imdb.images.set == 2);
% added the mean substruction per camera 
vggNet = load(fullfile(vl_rootnnPath, 'imagenet-vgg-verydeep-16.mat'));
meanImage = squeeze(vggNet.meta.normalization.averageImage);

allCertainty = zeros(numel(valInd), 2);
batchSize = 1;
for batchIdx = 1:batchSize:numel(valInd)
    disp(['batch index : ',num2str(batchIdx), '/', num2str(numel(valInd))]);
    tempImage = outData(:,:,:,valInd(batchIdx));
    tempImage(:,:,1) = tempImage(:,:,1) - meanImage(1);
    tempImage(:,:,2) = tempImage(:,:,2) - meanImage(2);
    tempImage(:,:,3) = tempImage(:,:,3) - meanImage(3);
    res1 = vl_simplenn(vggNet, tempImage);
    res = vl_simplenn(net, res1(35).x);
    squeeze(res(end).x)
%     [certainty,  predictions] = max(res(end).x, [], 3);
%     allCertainty(batchIdx) = certainty;
%     allPredictions(batchIdx) = predictions;
    allCertainty(batchIdx, :) = squeeze(squeeze(res(end).x));
end
[~, allPredictions] = max(allCertainty, [], 2);
groundTruth = imdb.images.labels(valInd);
groundTruth =groundTruth';

[confusionMat,order] = confusionmat(groundTruth,allPredictions);
for classInd = 1:8
    counter = 0;
    classImageIndices = find(imdb.images.labels(valInd) == classInd);
    for imageIdx = 1:numel(classImageIndices)
        if (imdb.images.labels(valInd(classImageIndices(imageIdx))) ~= allPredictions(classImageIndices(imageIdx)))
            allCertainty(classImageIndices(imageIdx), :)
            counter = counter + 1;
            imshow(outData(:,:,:,valInd(classImageIndices(imageIdx)))/255);
            title([num2str(allPredictions(classImageIndices(imageIdx))), '/', num2str(classInd), '   ', num2str(counter)]);
            waitforbuttonpress();
        end
    end
end



end

