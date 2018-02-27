# ISIC_melanoma

This is the repo for blog post <a href='http://yinniyu.github.io/posts/melanoma'>Binary Classifier for Melanoma Using MobileNet</a>. The Keras model runs on Tensorflow backend, and the AMI used was Bitfusion Ubuntu 14 TensorFlow with instance type of g2.2xlarge.

The general workflow is as follows: segmentation of original image --> mask image --> run model on masked image to predict.

<p align='center'><figure>
   <img src="graphics/mobilenet schematic.png" alt="mobilenet" >
   <figcaption>Fig.1. Schematic of the mobilenet architecture used in this repo.</figcaption>
</figure> 
