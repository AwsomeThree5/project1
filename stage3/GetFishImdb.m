function [ imdb ] = GetFishImdb(  )
%GETCARIMDB Summary of this function goes here
% Preapre the imdb structure, returns image data with mean image subtracted

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
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3'));
    CreateData();
end
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'data.mat'));
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

N = numel(data.label);
imagePaths = cell(N,1);
% calculate the width/height ratio for most of the images
if 1
    for imageIdx = 1:N
        if round(imageIdx/100) == imageIdx/100
            disp(['image number : ',num2str(imageIdx)]);
        end
        imagePaths{imageIdx} = fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx});
    end
end


w = 400;
h = w;

if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'outData.mat' ), 'file')
    outData = zeros(h,w,3,N, 'single');
    for imageIdx = 1:N
        try
            if round(imageIdx/100) == imageIdx/100
                disp(['image number : ',num2str(imageIdx)]);
            end
            img = im2single(imread(imagePaths{imageIdx}));
            % if grayscale
            if numel(size(img)) == 2
                img = cat(3, img, img, img);
            end
            cameraNumber = data.label{imageIdx};
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
            outData(:,:,:,imageIdx) = img(yMin+1:yMax, xMin+1:xMax , :);
        catch
            disp('waittt');
        end
    end
    save (fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'outData.mat' ), 'outData', '-v7.3');
end

load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3', 'outData.mat' ));
labels = [data.label{:}];

% find the train set indice
numOfTrain = min(find([data.set{:}] == 2)) - 1;
set(1:numOfTrain) = 1;
set(numOfTrain+1 : numel(data.set)) = 2;

% remove mean in any dimension
dataMean = mean(outData(:,:,:,find(set==1)), 4);
outData =  bsxfun(@minus, outData, dataMean);

% verify that values are between 0 and 1
if 0
    z = reshape(data,h*w,3,N);
    minZ = min(z);
    maxZ = max(z);
    z = bsxfun(@minus, z, minZ) ;
    z = bsxfun(@rdivide, z, maxZ-minZ);
    data = reshape(z, h, w, 3, N) ;
end


% initialize imdb parameters
imdb.meta.sets = {'train', 'val'};
imdb.meta.classes =unique(data.fishType);
imdb.images.labels = labels;
imdb.images.set = set;
imdb.images.data = outData;




end

