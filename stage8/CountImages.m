
folderPath = 'C:\Users\David\Desktop\YesNoFish\Fish';
dirStruct = dir([folderPath, '\*jpg']);
imageNames = {dirStruct.name}';
imageNames = cellfun(@(x) strsplit(x, {'_'}), imageNames(:), 'UniformOutput', false);
	