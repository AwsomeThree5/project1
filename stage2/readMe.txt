this part is an attempt to classify the different cameras
first, we need to construct a struct, containing all images in each cluster
each cluster is considered a 'label'.
we will randomly subset 80% of the images for train examples, and 20% for test data
the images will be resized to a reasonable size, keeping the original aspect ratio (or not?)

the net will be saved under 'data/fish/stage2' in the matconvnet path.