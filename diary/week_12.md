# Feat

Although ilustrationtovec is a bit of a misnomer, it is actually a feature extraction technique. And one of the key things i realised compared to how i was manually doing it
i found that there were significant changes and the process of extracting
features was very intensive for ilustrationtovec, such that there had to be core
refactoring, and muli-threading technique needed to be in place for this to be
succesfull, im still fixing the issues regarding this, Further more, i do
believe there May not be enough time to implement the previous models that i was
using to test and implement before hand.

# Observations

The use of auxiliary GANs in the ilustrationtovec model has provided better results compared to previous models. This is due to the improved ability of the auxiliary GANs to capture fine-grained details in the illustrations and generate more accurate vector representations.
But once again, much more resource intensive. Further more i will have to do
core optimization and refactoring for this to be succesfull.

# Self Observation and What Next

I had to ditch my previous work on previous feature extraction models as it was not providing the desired results. However, the previous work can still be useful in certain scenarios where the illustrations are not too complex. To further improve the performance of the current ilustrationtovec setup, I plan on exploring the use of different loss functions and implementing more advanced generator architectures. Additionally, I will be using various helper functions to fix a few issues I have encountered with the current setup. These modifications should help to further enhance the accuracy and effectiveness of the ilustrationtovec model.
