function [ ] = RunVggOnSelectiveBB( )
%RUNVGGONSELECTIVEBB Summary of this function goes here
%   Detailed explanation goes here

%FORWARDVGG Summary of this function goes here
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

currentFolder = fileparts(currentPath);
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'));
    CreateData();
end

% load the data
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'imdb.mat'));
% outData = h5read([vl_rootnnPath, '\data\fish-cnn-stage5\outData.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5']);
finalData = h5read([vl_rootnnPath, '\data\fish-cnn-stage5\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5']);
% h = size(outData,1);
% w = size(outData,2);

% load vgg-net
vggNet = load(fullfile(vl_rootnnPath, 'imagenet-vgg-verydeep-16.mat'));
outLayer = 35;
forwardNet.layers = vggNet.layers(35:end);

% imageIdx = 3000;
% meanImage = squeeze(vggNet.meta.normalization.averageImage);
% tempImage = outData(:,:,:,imageIdx);
% tempImage(:,:,1) = tempImage(:,:,1) - meanImage(1);
% tempImage(:,:,2) = tempImage(:,:,2) - meanImage(2);
% tempImage(:,:,3) = tempImage(:,:,3) - meanImage(3);
% res2 = vl_simplenn(vggNet, tempImage);
% difff = res2(outLayer).x - finalData(:,:,:,imageIdx);

fishClasses = [1,2,3,4,5,6,7,8,9,390,391,392,393,394,395,396,397];
predictions = zeros(size(finalData, 4), 2);
for fishThresh = linspace(0.80, 0.99, 8)
    for imageIdx = 1:size(finalData, 4)
        %     disp(['image number : ', num2str(imageIdx)]);
        res = vl_simplenn(forwardNet, finalData(:,:,:,imageIdx));
        softmax = res(end).x;
        [scores, class] = sort(softmax, 'descend');
        fishScore = 0;
        for fishClassIdx = 1:numel(fishClasses)
            fishScore = fishScore + scores(find(class == fishClasses(fishClassIdx)));
        end
        
        predictions(imageIdx, :) = [scores(1), class(1)];
        if fishScore > fishThresh && fishScore > scores(1)
            predictions(imageIdx, :) = [fishScore, 1];
        end
    end
    % save(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'predictions.mat'), 'predictions', '-v7.3');
    disp(['Thresh : ', num2str(fishThresh)]);
    groundTruth = [imdb.images.labels]';
    positiveTrue = find(groundTruth - predictions(:,2) == 0);
    disp([blanks(3),'positive recognition out of positive : ', num2str(numel(positiveTrue)/sum(groundTruth)),'   ',num2str(numel(positiveTrue)), '/', num2str(sum(groundTruth))]);
    negativeFalse = find(predictions(:,2) == 1 & groundTruth == 0);
    disp([blanks(3),'negative recognition out of negative: ', num2str(numel(negativeFalse)/numel(find(groundTruth == 0))), '    ',num2str(numel(negativeFalse)), '/', num2str(numel(find(groundTruth == 0)))]);
    imageNames = unique(imdb.images.name([positiveTrue ;negativeFalse]));
    disp([blanks(5), 'unique : ', num2str(numel(imageNames)/numel(unique(imdb.images.name))), '   ', num2str(numel(imageNames)), '/', num2str(numel(unique(imdb.images.name)))]);
    disp([blanks(5), 'percentage : ', num2str(numel(imageNames)/(numel(negativeFalse)+numel(positiveTrue))) ]);
    
end
end



