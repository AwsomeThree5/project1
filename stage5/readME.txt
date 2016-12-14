in this step we already have clustered images by camera view
we want to manually define bounding boxes for each of the cameras.
the fish should be in about the same place for each camera, therefor we believe that this boundingBox solution should work.

for each camera we will determine a center point for our bounding box. we will crop a fixed HxW size bounding box for each camera.
we will then show some example to the cropped images, hoping that it will make some sense