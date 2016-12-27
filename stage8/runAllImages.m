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
fishDataPath = 'C:\Users\David\Desktop\YesNoFish\Fish\';
realOutPath = 'C:\Users\David\Desktop\YesNoFish\RealFish';
noRealOutPath = 'C:\Users\David\Desktop\YesNoFish\NoRealFish';
mkdir(realOutPath);
mkdir(noRealOutPath);
% run on fish data end extract the camera number, based on 'data' struct
dirStruct = dir([fishDataPath, '\*jpg']);
imageNames = {dirStruct.name}';
imageNames = cellfun(@(x) [fishDataPath,x], imageNames(:), 'UniformOutput', false);

load('scores.mat')
for imageIdx = 1:numel(imageNames)
    disp(['image number : ', num2str(imageIdx), '/', num2str(numel(imageNames))]);
    img = imread(imageNames{imageIdx});
    [~,name,type] = fileparts(imageNames{imageIdx});
    if (scores(imageIdx,1) < 0.5)
       imwrite(img, fullfile(noRealOutPath, [name,type]));
    else
        imwrite(img, fullfile(realOutPath, [name,type]));
    end
end



%   useGpu: 0
%          numThreads: 12
%           imageSize: [224 224]
%            cropSize: 1
%     subtractAverage: [3x1 double]



    