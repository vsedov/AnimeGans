# Models

There should be 4 core models layouts listed in this folder


## DCGAN Varient One
### Works for the danbaru dataset
DCGANs (Deep Convolutional Generative Adversarial Networks) are a variant of GANs that use convolutional layers in both the generator and the discriminator. Convolutional layers are well suited for image generation tasks as they can learn spatial hierarchies of features from the input image. DCGANs have been used successfully in various image generation tasks, such as creating realistic images of faces, landscapes, and animals.

This also has the ability to work with multiple layers as well, which i found to be quite fruitful when developing the basic dataset.


## DCGAN Varient Two
### Works for the danbaru dataset
DCGANs with custom tags are a modification of the original DCGAN architecture that allows the generation of images with specific attributes or tags. This is achieved by incorporating the desired tags as additional input to the generator and discriminator. During the training process, the generator learns to generate images that match the desired tags while the discriminator tries to distinguish the generated images from the real images with the same tags.

## ACGANS
### Works for `crete_local_dataset`  only and will not work for the other dataset that is the custom one that you fetch from danburu
ACGANs with IlustrationVec (Auxiliary Classifier Generative Adversarial Networks with Illustration Vector) is a variant of GANs that incorporates class information in the form of an illustration vector. The illustration vector provides additional information to the generator about the class of the image it should generate. The generator is trained to not only generate realistic images, but also images that belong to a specific class. The discriminator is trained to not only distinguish the generated images from real images, but also to classify the generated images into the correct class. The outcome of the ACGANs with IlustrationVec training process is a generator that can produce images that are both realistic and belong to a specific class.

## CGAN model
### Works for `crete_local_dataset`  only and will not work for the other dataset that is the custom one that you fetch from danburu
This was not fully tested due to time constraints


# Use Case
In this file, There are several models, that fit to different use cases, the reason why this May be the case  is because i wanted to test
different architectures and different examples and usecases for generaeting images, optimization tools and more, most of the time, i found that opmization was a tad bit harder than originally anticipated but, the improvements from going from a basic cgan to acgans are prevelent within every repsect of the data, and their outputs.
