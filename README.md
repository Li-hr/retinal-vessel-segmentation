# retinal-vessel-segmentation
A segmentation project on retinal vessel based on two different ways: one is U-Net, the other one is based on 2D Gaussian Matched Filter
# DATASET INTRODUCTION
The photographs for the DRIVE database were obtained from a diabetic retinopathy screening program in The Netherlands. The screening population consisted of 400 diabetic subjects between 25-90 years of age. Forty photographs have been randomly selected, 33 do not show any sign of diabetic retinopathy and 7 show signs of mild early diabetic retinopathy. Here is a brief description of the abnormalities in these 7 cases:
The set of 40 images has been divided into a training and a test set, both containing 20 images. For the training images, a single manual segmentation of the vasculature is available. For the test cases, two manual segmentations are available; one is used as gold standard, the other one can be used to compare computer generated segmentations with those of an independent human observer. Furthermore, a mask image is available for every retinal image, indicating the region of interest. All human observers that manually segmented the vasculature were instructed and trained by an experienced ophthalmologist. They were asked to mark all pixels for which they were for at least 70% certain that they were vessel.

**The dataset download link**:https://drive.grand-challenge.org/

# Based on U-Net

## U-Net architecture

