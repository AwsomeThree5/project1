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
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'))
    mkdir(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5'));
    CreateData();
end
load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'data.mat'));
boundingBoxCentroid = load('boundingBoxCentroid.mat');
boundingBoxCentroid = boundingBoxCentroid.boundingBoxCentroid;

N = numel(data.label);
imagePaths = cell(N,1);
% calculate the number of boxes (good and bad) only for images with good
% boxes (which means that a fish was detected)
if 1
    numberOfBoxes = 0;
    for imageIdx = 1:N
        imagePaths{imageIdx} = fullfile(dataFolder, data.fishType{imageIdx}, data.name{imageIdx});
        if (~isempty(data.goodBoxes{imageIdx}))
            numberOfBoxes = numberOfBoxes + size(data.goodBoxes{imageIdx}, 1) + size(data.badBoxes{imageIdx}, 1);
        end
    end
end


w = 224;
h = w;

numberOfBoxes = 14943;
labels = zeros(1,numberOfBoxes);
counter = 0;
if ~exist(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage5', 'outData.h5' ), 'file')
    outData = zeros(h,w,3,numberOfBoxes, 'single');
    for imageIdx = 1:N
        try 
            if round(imageIdx/100) == imageIdx/100
                disp(['image number : ',num2str(imageIdx)]);
            end
            img = single(imread(imagePaths{imageIdx}));
            % if grayscale
            if numel(size(img)) == 2
                img = cat(3, img, img, img);
            end
            if (~isempty(data.goodBoxes{imageIdx}))
                % run through the good images
                for goodBoxIdx = 1:size(data.goodBoxes{imageIdx}, 1)
                    counter = counter + 1;
                    box = data.goodBoxes{imageIdx}(goodBoxIdx,:);
                    % later i would like to do something smarter here
                    outData(:,:,:,counter) = imresize(img(box(1):box(3),box(2):box(4), :), [h,w]);
                    labels(counter) = 1;
                    imdb.images.name{counter} = data.name{imageIdx};
                    imdb.images.box{counter} = box;
                    imdb.images.camera{counter} = data.label{imageIdx}; 
                    imdb.images.fishType{counter} = data.fishType{imageIdx};
                end
                % run through the bad images
                for badBoxIdx = 1:size(data.badBoxes{imageIdx}, 1)
                    counter = counter + 1;
                    box = data.badBoxes{imageIdx}(badBoxIdx,:);
                    % later i would like to do something smarter here
                    outData(:,:,:,counter) = imresize(img(box(1):box(3),box(2):box(4), :), [h,w]);
                    labels(counter) = 0;
                    imdb.images.name{counter} = data.name{imageIdx};
                    imdb.images.box{counter} = box;
                    imdb.images.camera{counter} = data.label{imageIdx}; 
                    imdb.images.fishType{counter} = data.fishType{imageIdx};
                end
            end
           
        catch
            disp('wrong box, throwing...');
            labels(counter) = -1;
        end 
    end
    
    wrongBoxes = find(labels == -1);
    labels(wrongBoxes) = [];
    imdb.images.name(wrongBoxes) = [];
    imdb.images.box(wrongBoxes) = [];
    imdb.images.camera(wrongBoxes) = [];
    imdb.images.fishType(wrongBoxes) = [];
    outData(:,:,:,wrongBoxes) = [];
    numberOfBoxes = numel(labels);
    
    rng(1)
    randInd = randperm(numberOfBoxes);
    imdb.images.name = imdb.images.name(randInd);
    imdb.images.box = imdb.images.box(randInd);
    imdb.images.camera = imdb.images.camera(randInd);
    imdb.images.fishType = imdb.images.fishType(randInd);
    labels = labels(randInd);
    outData = outData(:,:,:,randInd);
    h5create([vl_rootnnPath, '\data\fish-cnn-stage5\outData.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5'], size(outData),'Datatype', 'single');
    h5write([vl_rootnnPath, '\data\fish-cnn-stage5\outData.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5'],outData);
%     clear outData
end

save labels labels
outData = h5read([vl_rootnnPath, '\data\fish-cnn-stage5\outData.h5'], ['/',vl_rootnnPath,'\data\fish-cnn-stage5']);

% find the train set indice
numOfTrain = floor(0.8 * numberOfBoxes);
set(1:numOfTrain) = 1;
set(numOfTrain+1 : numberOfBoxes) = 2;

% remove mean in any dimension
% dataMean = zeros(h,w,3,max([data.camera{:}]));
% for cameraIdx = 1:max([data.camera{:}])
%     indices = find([data.camera{1:numOfTrain}] == cameraIdx);
%     disp(['camera : ',num2str(cameraIdx), '... numImages : ',num2str(numel(indices)), '/', num2str(numOfTrain)]);
%     dataMean(:,:,:,cameraIdx) = mean(outData(:,:,:,indices), 4);
% end
% % dataMean = mean(outData(:,:,:,1:numOfTrain), 4);
% save dataMean dataMean
% outData =  bsxfun(@minus, outData, dataMean);

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
% imdb.images.data = outData;
% imdb.images.name = data.name;
% imdb.images.camera = data.camera;




end

