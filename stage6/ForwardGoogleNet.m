function [ ] = ForwardGoogleNet(  )
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

% load Google-Net 
net = dagnn.DagNN.loadobj(load(fullfile(vl_rootnnPath, 'imagenet-googlenet-dag.mat'))) ;


fishClasses = [1,2,3,4,5,6,7,8,9,390,391,392,393,394,395,396,397];
for imageIdx = 1: numel(data.name)
    disp(['image number : ',num2str(imageIdx), '/', num2str(numel(data.name))]);
    %     tempImage = outData(:,:,:,imageIdx);
    tempImage = single(imread(fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx})));
    %resizing the image to hald the size in each dimension
    tempImage = imresize(tempImage,0.5, 'bicubic');
    resizedMeanImage = imresize(net.meta.normalization.averageImage, [size(tempImage,1), size(tempImage,2)]);
    tempImage(:,:,1) = tempImage(:,:,1) - resizedMeanImage(:,:,1);
    tempImage(:,:,2) = tempImage(:,:,2) - resizedMeanImage(:,:,1);
    tempImage(:,:,3) = tempImage(:,:,3) - resizedMeanImage(:,:,1);
    tic ;net.eval({'data', tempImage});toc
    scores = squeeze(gather(net.vars(end).value));
    fishScoreMat = scores(:,:,fishClasses);
    fishScoreMat = sum(fishScoreMat, 3);
    scoreImage = uint8(fishScoreMat*255);
    scoreImage = imresize(scoreImage,2,'bicubic');
    mkdir(fullfile(dataFolder,'smallScore-GoogleNet', data.fishType{imageIdx}));
    imageName = strsplit(data.name{imageIdx}, '.');
    imwrite(scoreImage,fullfile(dataFolder,'smallScore-GoogleNet', data.fishType{imageIdx}, [imageName{1},'.bmp']));
end

% h5create([vl_rootnnPath, '\data\fish-cnn-stage6\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage6'], size(finalData),'Datatype', 'single');
% h5write([vl_rootnnPath, '\data\fish-cnn-stage6\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage6'],finalData);

end

