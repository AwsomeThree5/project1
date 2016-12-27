function [boxes] = RunSelectiveSearch(img,xMin,xMax,yMin,yMax )
%RUNSELECTIVESEARCH Summary of this function goes here
%   Detailed explanation goes here
% This demo shows how to use the software described in our IJCV paper:
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation.
minSize = k;
sigma = 0.8;


h = yMax-yMin;
w = xMax-xMin;
centroidBox = [yMin, xMin, h, w];
% Perform Selective Search
[boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(img, sigma, k, minSize, colorType, simFunctionHandles);
disp([blanks(2),'found : ',num2str(size(boxes,1)), ' boxes']);
% built in function
boxes = BoxRemoveDuplicates(boxes);
disp([blanks(2),'filtered duplicates: ',num2str(size(boxes,1)), ' boxes']);
% my function of removing additional boxes
boxes = FilterIrrelevantBoxes(boxes, centroidBox, h, w, img);
% disp([blanks(2),'final : ',num2str(size(boxes,1)), ' boxes']);
% Show boxes
% ShowRectsWithinImage(boxes, 3, 3, img);
%         title(data.fishType{imageIdx});
%         if ~isempty(data.bbAnnotation{imageIdx})
%             goodBoxes = [];
%             badBoxes = [];
%             tempBoxes = boxes;
%             tempBoxes(:,3) = tempBoxes(:,3) - tempBoxes(:,1)+1;
%             tempBoxes(:,4) = tempBoxes(:,4) - tempBoxes(:,2)+1;
%             for bbInd = 1:size(data.bbAnnotation{imageIdx}, 1)
%                 [value, ind] = sort(bboxOverlapRatio(tempBoxes, data.bbAnnotation{imageIdx}(bbInd, :), 'Min'), 'descend');
%                 goodBoxes = cat(1, goodBoxes, boxes(ind(value > 0.4), :));
%                 badBoxes = cat(1,badBoxes, boxes(ind(value <= 0.4), :));
% %                 figure;imshow(img(data.bbAnnotation{imageIdx}(bbInd,1):data.bbAnnotation{imageIdx}(bbInd,1)+data.bbAnnotation{imageIdx}(bbInd,3),...
% %                               data.bbAnnotation{imageIdx}(bbInd,2):data.bbAnnotation{imageIdx}(bbInd,2)+data.bbAnnotation{imageIdx}(bbInd,4), :));
%             end
%             % sort out duplicate boxes
%             [~, ind] = unique(goodBoxes, 'rows');
%             goodBoxes = goodBoxes(ind, :);
%
%             badBoxes = boxes;
%             isBadBox = ones(size(badBoxes,1), 1);
%             for boxIdx = 1:size(badBoxes,1)
%                 for goodBoxIdx = 1:size(goodBoxes, 1)
%                     if isequal(badBoxes(boxIdx, :), goodBoxes(goodBoxIdx, :))
%                         isBadBox(boxIdx) = 0;
%                     end
%                 end
%             end
%             badBoxes = badBoxes(logical(isBadBox), :);
% %             ShowRectsWithinImage(newBoxes, 3, 3, img)
%             if (size(goodBoxes,1) + size(badBoxes,1) ~= size(boxes,1))
%                 disp('WRONGGGG');
%             end
%             data.goodBoxes{clusteredImagesInd(imageIdx)} = goodBoxes;
%             data.badBoxes{clusteredImagesInd(imageIdx)} = badBoxes;
%             if (~isempty(goodBoxes))
%                 counter = counter + 1;
%
%             end
%         end


end

