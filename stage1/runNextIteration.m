function [ output_args ] = runNextIteration( )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
dataFolder = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish\train\train';
dataFolder = [dataFolder, '\ALB'];

% find images names inside the data folder
allImages = dir([dataFolder,  '\*.jpg']);
allImages = {allImages(:).name}';

% find cluster folders
clusterNames = dir(dataFolder);
clusterNames(~[clusterNames.isdir]) = [];
clusterNames = {clusterNames(:).name}';
clusterNames(ismember(clusterNames, {'.', '..'})) = [];

%run through cluster folders, and generate a vector containing all
%clustered images names
clusteredImages =  cell(0,1);
counter = 0;
numOfClusters = numel(clusterNames);
for clusterIdx = 1:numOfClusters
    disp(['cluster number : ', num2str(clusterIdx), '/', num2str(numOfClusters)]);
    clusterFolder = [dataFolder, '\', clusterNames{clusterIdx}];
    clusterImages = dir([clusterFolder, '\*.jpg']);
    clusterImages = {clusterImages.name}';
    counter = counter + numel(clusterImages);
    clusteredImages(end+1:end+numel(clusterImages)) = clusterImages;
end

% find which images are unclustered
[imagePaths,~,uniqueNames] = unique([allImages ;clusteredImages]);
[histCounts, ~] = hist(uniqueNames, numel(imagePaths));
unclusteredImagesIndices = find(histCounts == 1);
unclusteredImages = imagePaths(unclusteredImagesIndices);
%sanity check
disp('sanity check : ');
disp([blanks(3),'number of clustered : ', num2str(numel(clusteredImages))]);
disp([blanks(3),'number of unclustered : ', num2str(numel(unclusteredImagesIndices))]);
disp([blanks(3),'Total : ', num2str(numel(unclusteredImagesIndices)+numel(clusteredImages)), '/', num2str(numel(allImages))]);

% run OpticalFlow
h = 200;
images.path = cell(numel(unclusteredImages),1);
warning off;
for imageIdx = 15:numel(unclusteredImages)
    disp(['image number : ', num2str(imageIdx), '/', num2str(numel(unclusteredImages))]);
    img = imread(fullfile(dataFolder, unclusteredImages{imageIdx}));
    img = imresize(img, [h,h]);
    figure;
    subplot(1,2,1)
    imshow(img);
    similarityMatrix = zeros(1,numOfClusters);
    for clusterIdx = 1:numOfClusters
        disp([blanks(4), 'cluster number : ', num2str(clusterIdx), '/', num2str(numOfClusters)]);
        clusterFolder = [dataFolder, '\', clusterNames{clusterIdx}];
        clusterImages = dir([clusterFolder, '\*.jpg']);
        clusterImages = {clusterImages.name}';
        randInd = randperm(numel(clusterImages));
        numOfImagesToRunOn = min(1, numel(clusterImages));
        similarity = zeros(1,numOfImagesToRunOn);
        for imgIdx = 1:numOfImagesToRunOn
            clusterImg = imread(fullfile(clusterFolder,clusterImages{randInd(imgIdx)}));
            clusterImg = imresize(clusterImg, [h,h]);
            subplot(1,2,2);
            imshow(clusterImg);
            similarity(imgIdx) = mean(mean(mean(abs(RunSIFTFlow(rgb2gray(img),rgb2gray(clusterImg))))));
        end
        similarityMatrix(clusterIdx) = mean(similarity);
    end
    similarityMatrix
    [num,chosenCluster] = min(similarityMatrix);
    clusterFolder = [dataFolder, '\', clusterNames{chosenCluster}];
    clusterImages = dir([clusterFolder, '\*.jpg']);
    clusterImages = {clusterImages.name}';
    clusterImg = imread(fullfile(clusterFolder,clusterImages{1}));
    clusterImg = imresize(clusterImg, [h,h]);
    subplot(1,2,2);
    imshow(clusterImg);
    if num < 6
        title('TRUE')
    else
        title('FALSE')
    end
    drawnow;
end



% % calculate KMeans
% k = 15;
% [label, centroid, dis] = fkmeans(images.data, k);
% centroid = centroid';
% 
% % this are the mean maps for each class
% centroidMaps = reshape(centroid, [h,h,k]);
% for classIdx = 1:k
%     figure;
%     imshow(centroidMaps(:,:,classIdx), [])
%     title(['class number : ',num2str(classIdx)]);
% end
% 
% desiredClass = 3;
% for imageIdx = 1:size(images.data, 1)
%     if label(imageIdx) == desiredClass
%         imshow(reshape(images.data(imageIdx, :), [h,h]), []);
%         waitforbuttonpress();
%     end
% end
end

function [flow] = RunOpticalFlow(im1,im2)
addpath('mex');

% we provide two sequences "car" and "table"
% example = 'table';
%example = 'car';

% load the two frames
im1 = im2double(im1);
im2 = im2double(im2);

% im1 = imresize(im1,0.5,'bicubic');
% im2 = imresize(im2,0.5,'bicubic');

% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation
tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc

figure;imshow(im1);figure;imshow(warpI2);



% % output gif
% clear volume;
% volume(:,:,:,1) = im1;
% volume(:,:,:,2) = im2;
% if exist('output','dir')~=7
%     mkdir('output');
% end
% frame2gif(volume,fullfile('output',[example '_input.gif']));
% volume(:,:,:,2) = warpI2;
% frame2gif(volume,fullfile('output',[example '_warp.gif']));


% visualize flow field
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
imflow = flowToColor(flow);

% figure;imshow(imflow);
% imwrite(imflow,fullfile('output',[example '_flow.jpg']),'quality',100);



end

function [flow] = RunSIFTFlow(im1,im2)

im1=im2double(im1);
im2=im2double(im2);


% Step 2. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

Sift1=dense_sift(im1,patchsize,gridspacing);
Sift2=dense_sift(im2,patchsize,gridspacing);

% visualize the SIFT image
% figure;imshow(showColorSIFT(Sift1));title('SIFT image 1');
% figure;imshow(showColorSIFT(Sift2));title('SIFT image 2');

% Step 3. SIFT flow matching

% prepare the parameters
SIFTflowpara.alpha=2;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

tic;[vx,vy,energylist]=SIFTflowc2f(Sift1,Sift2,SIFTflowpara);toc
flow(:,:,1) = vx;
flow(:,:,2) = vy;


end