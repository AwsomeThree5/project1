function [ ] = GatherHeadAndTailAnnotation()
%GATHERHEADANDTAILANNOTATION Summary of this function goes here

%   these links are from an israeli guy who claimed to annotate the head
%   and tail of fish. he was wrong. his bounding box are full of shit.
%   instead I used a different guy's annotations
% yftLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5373/yft_labels.json');
% sharkLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5374/shark_labels.json');
% lagLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5375/lag_labels.json');
% dolLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5376/dol_labels.json');
% betLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5377/bet_labels.json');
% albLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5378/alb_labels.json');

yftLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5461/yft_labels.json');
sharkLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5459/shark_labels.json');
lagLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5463/lag_labels.json');
dolLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5460/dol_labels.json');
betLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5458/bet_labels.json');
albLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5462/alb_labels.json');
otherLabels = webread('https://www.kaggle.com/blobs/download/forum-message-attachment-files/5471/other_labels.json');


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

% load struct of images indicating the path and cluster.
data = load(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'));
data = data.data;

for imageIdx = 1:numel(data.label)
    
    disp(['image number : ', num2str(imageIdx), '/', num2str(numel(data.label))]);
    annotation = [];
    data.bbAnnotation{imageIdx} = [];
    
    if strcmpi(data.fishType{imageIdx}, 'alb')
        if ~isempty(albLabels(strcmp(data.name{imageIdx}, {albLabels.filename}')))
            annotation = albLabels(strcmp(data.name{imageIdx}, {albLabels.filename}')).annotations;
        end
        
    elseif strcmpi(data.fishType{imageIdx}, 'bet')
        if ~isempty(betLabels(strcmp(data.name{imageIdx}, {betLabels.filename}')))
            annotation = betLabels(strcmp(data.name{imageIdx}, {betLabels.filename}')).annotations;
        end
        
    elseif strcmpi(data.fishType{imageIdx}, 'dol')
        if ~isempty(dolLabels(strcmp(data.name{imageIdx}, {dolLabels.filename}')))
            annotation = dolLabels(strcmp(data.name{imageIdx}, {dolLabels.filename}')).annotations;
        end
        
    elseif strcmpi(data.fishType{imageIdx}, 'lag')
        if ~isempty(lagLabels(strcmp(data.name{imageIdx}, {lagLabels.filename}')))
            annotation = lagLabels(strcmp(data.name{imageIdx}, {lagLabels.filename}')).annotations;
        end
        
    elseif strcmpi(data.fishType{imageIdx}, 'nof')
        % there is no fish by definition
        
    elseif strcmpi(data.fishType{imageIdx}, 'other')
        if ~isempty(otherLabels(strcmp(data.name{imageIdx}, {otherLabels.filename}')))
            annotation = otherLabels(strcmp(data.name{imageIdx}, {otherLabels.filename}')).annotations;
        end
    elseif strcmpi(data.fishType{imageIdx}, 'shark')
        if ~isempty(sharkLabels(strcmp(data.name{imageIdx}, {sharkLabels.filename}')))
            annotation = sharkLabels(strcmp(data.name{imageIdx}, {sharkLabels.filename}')).annotations;
        end
        
    elseif strcmpi(data.fishType{imageIdx}, 'yft')
        if ~isempty(yftLabels(strcmp(data.name{imageIdx}, {yftLabels.filename}')))
            annotation = yftLabels(strcmp(data.name{imageIdx}, {yftLabels.filename}')).annotations;
        end
    end
   
    
    if ~isempty(annotation)
        % there may be more than one bounding box
        for bbInd = 1:numel(annotation)
            currentBox = annotation(bbInd);
            data.bbAnnotation{imageIdx} = cat(1, data.bbAnnotation{imageIdx}, [currentBox.y, currentBox.x, currentBox.height, currentBox.width]);
        end
    end
end

% save struct of images with the boundingBoxAnnotations
save(fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage2', 'data.mat'), 'data');


end

