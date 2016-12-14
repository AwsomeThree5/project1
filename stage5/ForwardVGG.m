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
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'));
    CreateData();
end

% load the data
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'imdb.mat'));
outData = h5read([vl_rootnnPath, '\data\fish-cnn-stage5\outData.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5']);
h = size(outData,1);
w = size(outData,2);

% load vgg-net
net = load(fullfile(vl_rootnnPath, 'imagenet-vgg-verydeep-16.mat'));

outLayer = 35;
if h == 224
    outLayer = 35;
end

finalData = zeros(1,1,4096,size(outData,4), 'single');
meanImage = squeeze(net.meta.normalization.averageImage);
for imageIdx = 207:size(outData,4)
    disp(['image number : ',num2str(imageIdx), '/', num2str(size(outData,4))]);
%     tempImage = outData(:,:,:,imageIdx);
    tempImage = single(imread('C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish\train\train\ALB\5\img_00090.jpg'));
%     subplot(2,1,1)
    imshow(tempImage/255)
%     subplot(2,1,2)
%     imageName = imdb.images.name{imageIdx};
%     box = imdb.images.box{imageIdx};
%     imagePath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish\train\train\ALB\5\img_06282.jpg';
%     fullImage = imread(imagePath);
%     imshow(fullImage(box(1):box(3),box(2):box(4), :));
%     
    tempImage(:,:,1) = tempImage(:,:,1) - meanImage(1);
    tempImage(:,:,2) = tempImage(:,:,2) - meanImage(2);
    tempImage(:,:,3) = tempImage(:,:,3) - meanImage(3);
    tic;res = vl_simplenn(net, tempImage);toc
    finalData(:,:,:,imageIdx) = res(outLayer).x;
    scores = res(end).x;
    [score, class] = sort(scores, 'descend');
    disp(['top : ',num2str(score(1)), ' class : ',net.meta.classes.description{class(1)}, '(', num2str(class(1)), ') ', ]);
    fishClasses = [1,2,3,4,5,6,7,8,9,390,391,392,393,394,395,396,397];
    fishScore = zeros(size(score,1), size(score,2));
    
    fishScoreMat = scores(:,:,fishClasses);
    fishScoreMat = sum(fishScoreMat, 3);
    figure;imhist(fishScoreMat)
        figure; imshow(imresize(fishScoreMat, [size(tempImage,1), size(tempImage,2)]), [])
        
        
    
    disp(['fish score : ', num2str(fishScore)]);
    waitforbuttonpress();
end

% h5create([vl_rootnnPath, '\data\fish-cnn-stage5\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5'], size(finalData),'Datatype', 'single');
% h5write([vl_rootnnPath, '\data\fish-cnn-stage5\finalDataNormalized.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5'],finalData);

end


