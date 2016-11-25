
% initialize paths
vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
fishType = 'ALB';
projectName = 'project1';

addpath(genpath(vl_rootnnPath))
try
    warning off;
    vl_setupnn();   
catch
    disp('you need to initialize the matconvnet path first');
end
dataFolder = fullfile(vl_rootnnPath, 'data\fish\train\train');
dataFolder = [dataFolder, '\', fishType];
if ~exist(dataFolder, 'dir')
    disp('fish data path should look like : ');
    disp([blanks(3),'...vl_rootnn\data\fish\train\train']);
end

% include folders and subfolders of this project
currentPath = mfilename('fullpath');
gitPath = currentPath(1:strfind(currentPath, projectName)+numel(projectName)-1);
addpath(genpath(gitPath))

%lets start
imageNames = dir([dataFolder,  '\*.jpg']);
%height and width of the resized images
h = 400;
images.data = zeros(numel(imageNames), h*h);
images.path = cell(numel(imageNames),1);
for imageIdx = 1:numel(imageNames)
    disp(['image number : ', num2str(imageIdx)]);
    img = imread(fullfile(dataFolder, imageNames(imageIdx).name));
    img = rgb2gray(img);
    img = imresize(img, [h,h]);
    % proved to work better without these lines
%     img = img - mean(img(:));
%     img = img / std(img(:));
    images.data(imageIdx, :) = reshape(img, [h*h, 1]);
    images.path{imageIdx} = [dataFolder, '\', imageNames(imageIdx).name];
end

save images images '-v7.3'
% load ('images.mat');

% calculate KMeans
k = 15; %you can play with that...
[label, centroid, dis] = fkmeans(images.data, k);
centroid = centroid';

% this are the mean maps for each class
centroidMaps = reshape(centroid, [h,h,k]);
for classIdx = 1:k
    figure;
    imshow(centroidMaps(:,:,classIdx), [])
    title(['class number : ',num2str(classIdx)]);
end

% rewrite images to the cluster dirs
% BECAREFULL NOT TO OVERWRITE IN CASE YOU HAVE RUN THIS CODE ONCE BEFORE
for classIdx = 1:k
    disp(['class number : ',num2str(classIdx), '/', num2str(k)]);
    path = [dataFolder, '\', num2str(classIdx)];
    imageIndicesPerClass = find(label == classIdx);
    if exist(path, 'dir')
%         rmdir(path, 's');
    end
    mkdir(path);
    for imageIdx = 1 : numel(imageIndicesPerClass)
        disp([blanks(3), 'image number : ',num2str(imageIdx), '/', num2str(numel(imageIndicesPerClass))]);
        image = imread(fullfile(dataFolder, imageNames(imageIndicesPerClass(imageIdx)).name));
        imwrite(image, fullfile(path, imageNames(imageIndicesPerClass(imageIdx)).name));
    end
    
end


% display all images from a desiredClass
% desiredClass = 4;
% for imageIdx = 1:size(images.data, 1)
%     if label(imageIdx) == desiredClass
%         imshow(reshape(images.data(imageIdx, :), [h,h]), []);
%         waitforbuttonpress();
%     end
% end




