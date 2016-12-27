vl_rootnnPath = 'C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23';
desktopPath = 'C:\Users\David\Desktop';
projectName = 'project1';

addpath(genpath(vl_rootnnPath))
try
    warning off;
    vl_setupnn();   
catch
    disp('you need to initialize the matconvnet path first');
    return;
end
fishDataPath = 'C:\Users\David\Desktop\YesNoFish\nof\';
% run on fish data end extract the camera number, based on 'data' struct
dirStruct = dir([fishDataPath, '\*jpg']);
imageNames = {dirStruct.name}';
imageNames = cellfun(@(x) [fishDataPath,x], imageNames(:), 'UniformOutput', false);

load ('C:\Users\David\Desktop\matlab\matconvnet\matconvnet-1.0-beta23\matconvnet-1.0-beta23\data\fish-cnn-stage7\net-epoch-6.mat');
% load('net.mat')
net.layers{end}.type = 'softmax';
opts.useGpu =  0;
opts.numThreads = 12;
opts.imageSize = [224 224];
opts.cropSize = 1;
opts.subtractAverage = net.meta.normalization.averageImage;

predictions = -1 * ones(numel(imageNames),1);
scores = zeros(numel(imageNames), 2);
for imageIdx = 1:100:numel(imageNames)
    res = [];
    disp(['image : ', num2str(imageIdx),'/',num2str(numel(imageNames))]);
    batchSize = min(99,numel(imageNames)-imageIdx);
    data = getImageBatch(imageNames(imageIdx:imageIdx+batchSize), opts);%, 'prefetch', nargout == 0) ;
    res = vl_simplenn(net, data, [],[],'mode', 'test', 'conserveMemory', true);
    scores(imageIdx:imageIdx+batchSize, :) = squeeze(res(end).x)';
%     scores(imageIdx+batchSize, :)
    [~,predictions(imageIdx:imageIdx+batchSize)] = max(squeeze(res(end).x));
    predictions(imageIdx:imageIdx+batchSize) = abs(predictions(imageIdx:imageIdx+batchSize)-2);
    disp([blanks(3),'num of fish : ', num2str(numel(find(predictions(imageIdx:imageIdx+batchSize) == 1)))])
end
save scores1 scores '-v7.3'
save prediction1 prediction '-v7.3'


%   useGpu: 0
%          numThreads: 12
%           imageSize: [224 224]
%            cropSize: 1
%     subtractAverage: [3x1 double]



    