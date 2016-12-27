function data = getImageBatch(imagePaths, varargin)
% GETIMAGEBATCH  Load and jitter a batch of images

opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 1 ;

opts.imageSize = [227, 227] ;
opts.cropSize = 256 / 256 ;
opts.keepAspect = false ;
opts.subtractAverage = [] ;

opts.jitterFlip = false ;
opts.jitterLocation = false ;
opts.jitterAspect = 0 ;
opts.jitterScale = 0 ;
opts.jitterBrightness = 0 ;
opts.jitterContrast = 0 ;
opts.jitterSaturation = 0 ;

opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bicubic', ...
           'Resize', opts.imageSize(1:2), ...
           'CropSize', 1, ...
           'CropAnisotropy', 0, ...
           'Brightness', opts.jitterBrightness, ...
           'Contrast', opts.jitterContrast, ...
           'Saturation', opts.jitterSaturation} ;

if ~opts.keepAspect
  % Squashign effect
  args{end+1} = {'CropAnisotropy', 0} ;
end

if opts.jitterFlip
  args{end+1} = {'Flip'} ;
end

if opts.jitterLocation
  args{end+1} = {'CropLocation', 'random'} ;
else
%   args{end+1} = {'CropLocation', 'center'} ;
end

if opts.useGpu
  args{end+1} = {'Gpu'} ;
end

if ~isempty(opts.subtractAverage)
  args{end+1} = {'SubtractAverage', opts.subtractAverage} ;
end

args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg(args{:}) ;
  data = data{1} ;
end
