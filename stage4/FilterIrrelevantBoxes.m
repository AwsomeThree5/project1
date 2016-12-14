function [ boxes ] = FilterIrrelevantBoxes( boxes, centroidBox, h, w, img)
%FILTERIRRELEVANTBOXES Summary of this function goes here
%   filter boxes according to common sense and spatial centroidBox
%   information

goodBoxes = [];
for boxIdx = 1:size(boxes, 1)
    currentBox = boxes(boxIdx, :);
    currentBox(3) = currentBox(3) - currentBox(1)+1;
    currentBox(4) = currentBox(4) - currentBox(2)+1;
    
    boxSize = currentBox(3) * currentBox(4);
    % if the boxSize is too big (notice that a small box is acceptable)
    if boxSize > 0.5*h*w 
        continue;
    % if the box's height/width ratio is too skewed
    elseif currentBox(3)/currentBox(4) > 1.7 || currentBox(4)/currentBox(3) > 4
        continue;
    % if the box is almost one dimensional
    elseif currentBox(4) < 30 || currentBox(3) < 30
        continue;
    elseif bboxOverlapRatio(currentBox, centroidBox, 'Min') < 0.3
        continue;
    else
        goodBoxes = [goodBoxes, boxIdx];
    end
end
boxes = boxes(goodBoxes, :);
tempBoxes = boxes;
tempBoxes(:,3) = tempBoxes(:,3) - tempBoxes(:,1)+1;
tempBoxes(:,4) = tempBoxes(:,4) - tempBoxes(:,2)+1;

goodBoxes = ones(size(boxes, 1), 1);
for firstBoxIdx = 1:size(boxes, 1)
%     disp(['first : ', num2str(firstBoxIdx)]);
    for secondBoxIdx = 1:size(boxes, 1)
        if secondBoxIdx == firstBoxIdx
            continue;
        end
        overLap = bboxOverlapRatio(tempBoxes(firstBoxIdx,:), tempBoxes(secondBoxIdx,:), 'Min');
        firstBoxSize = tempBoxes(firstBoxIdx,3) * tempBoxes(firstBoxIdx,4);
        secondBoxSize = tempBoxes(secondBoxIdx,3) * tempBoxes(secondBoxIdx,4);
        if overLap > 0.5 && (firstBoxSize < secondBoxSize) %s&& goodBoxes(secondBoxIdx)
            goodBoxes(firstBoxIdx) = 0;
%             disp([blanks(3), 'second : ', num2str(secondBoxIdx)]);
            break;
        end
        
    end
end
boxes = boxes(logical(goodBoxes), :);
end

