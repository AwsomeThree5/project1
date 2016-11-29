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
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2'));
    CreateData();
end
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'));

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

if 0
    ratio = zeros([N, 1]);
    height = zeros([N, 1]);
    for imageIdx = 1:N
        if round(imageIdx/100) == imageIdx/100
            disp(['image number : ',num2str(imageIdx)]);
        end
        [h,w,~] = size(imread(imagePaths{imageIdx}));
        ratio(imageIdx) = h/w;
        height(imageIdx) = h;
        
    end
end


% determine the width and height of all images
finalRatio = 0.57; % = mean(ratio)
w = 100;
h = floor(w*finalRatio);

if 0
    outData = single(zeros(h,w,3,N));
    for imageIdx = 1:N
        if round(imageIdx/100) == imageIdx/100
            disp(['image number : ',num2str(imageIdx)]);
        end
        img = im2single(imread(imagePaths{imageIdx}));
        % if grayscale
        if numel(size(img)) == 2
            img = cat(3, img, img, img);
        end
        outData(:,:,:,imageIdx) = imresize(img, [h, w]);
    end
    save (fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'outData.mat' ), 'outData', '-v7.3');
end

load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'outData.mat' ));
labels = [data.label{:}];

% find the train set indice
numOfTrain = min(find([data.set{:}] == 2)) - 1;
set(1:numOfTrain) = 1;
set(numOfTrain+1 : numel(data.set)) = 2;

% remove mean in any dimension
dataMean = mean(outData(:,:,:,set), 4);
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
imdb.meta.classes = num2cell(1:max(max([labels(:)])));
imdb.meta.classes =imdb.meta.classes';
imdb.images.labels = labels;
imdb.images.set = set;
imdb.images.data = outData;







end

