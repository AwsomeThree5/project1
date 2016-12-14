function [ ] = ForwardVGG(  )
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
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage6'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage6'));
    CreateData();
end

% load the data
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'data.mat'));

% load vgg-net
net = load(fullfile(vl_rootnnPath, 'imagenet-vgg-verydeep-16.mat'));

meanImage = squeeze(net.meta.normalization.averageImage);
fishClasses = [1,2,3,4,5,6,7,8,9,390,391,392,393,394,395,396,397];
for imageIdx = 108:numel(data.name)
    disp(['image number : ',num2str(imageIdx), '/', num2str(numel(data.name))]);
%     tempImage = outData(:,:,:,imageIdx);
    tempImage = single(imread(fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx})));
    tempImage(:,:,1) = tempImage(:,:,1) - meanImage(1);
    tempImage(:,:,2) = tempImage(:,:,2) - meanImage(2);
    tempImage(:,:,3) = tempImage(:,:,3) - meanImage(3);
    tic;res = vl_simplenn(net, tempImage);toc
    scores = res(end).x;
    fishScoreMat = scores(:,:,fishClasses);
    fishScoreMat = sum(fishScoreMat, 3);
    scoreImage = uint8(fishScoreMat*255);
    mkdir(fullfile(dataFolder,'scoreImages', data.fishType{imageIdx}));
    imageName = strsplit(data.name{imageIdx}, '.');
    imwrite(scoreImage,fullfile(dataFolder,'scoreImages', data.fishType{imageIdx}, [imageName{1},'.bmp']));
end

% h5create([vl_rootnnPath, '\data\fish-cnn-stage6\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage6'], size(finalData),'Datatype', 'single');
% h5write([vl_rootnnPath, '\data\fish-cnn-stage6\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage6'],finalData);

end


