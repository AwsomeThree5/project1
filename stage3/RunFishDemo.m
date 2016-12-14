function [net, info] = RunFishDemo( varargin )
% THIS IS FOR STAGE 3 - FISH CLASSIFIER
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

opts.modelType = 'cnn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnnPath, 'data', 'fish-cnn-stage3') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data','fish') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.outDataPath = fullfile(opts.expDir, 'outData.h5');
opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

% h5create('C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage3\outData.h5', ['/','C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage3'], size(outData),'Datatype', 'single')
% h5write('C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage3\outData.h5', ['/','C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage3'],outData)

substructMean = 1;
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    imdb.images.data = h5read(opts.outDataPath, ['/',opts.expDir]);
    dataMean = load('dataMean');
    dataMean = dataMean.dataMean;
    if substructMean 
        for cameraIdx = 1:max([imdb.images.camera{:}])
            indices = find([imdb.images.camera{:}] == cameraIdx);
            disp(['camera : ',num2str(cameraIdx), '... numImages : ',num2str(numel(indices)), '/', num2str(numel(imdb.images.camera))]);
            imdb.images.data(:,:,:,indices) = bsxfun(@minus, imdb.images.data(:,:,:,indices), dataMean(:,:,:,cameraIdx));
        end     
    end
%     imdb.images.data = bsxfun(@minus, outData, dataMean);
else
    imdb = GetFishImdb() ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
    imdb.images.data = h5read(opts.outDataPath, ['/',opts.expDir]);
    load('dataMean');
end

switch opts.modelType
    case 'cnn'
        net = fish_init(size(imdb.images.data, 1),size(imdb.images.data, 2),size(imdb.images.data, 3)) ;
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainfn = @cnn_train ;
    case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 2)) ;
end

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end
end
% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch);
if rand > 0.5
    images = permute(images, [2,1,3,4]);
end
if rand > 0.5
    images = flipud(images);
end
if rand > 0.5
    images = permute(images, [2,1,3,4]);
end
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
end


