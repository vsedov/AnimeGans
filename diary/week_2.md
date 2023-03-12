# Feats

- Prior from last week, I had a deep dive into calculus, reading gilbert strangs book on calc and linear algebra to
catch my lost knolwedge gaps, i also took the time and effort to speak to some senior machine learning engineers to
guide me and give me guidance regarding how and what steps to do next.

  - Majority of them did state that for generative networks, understadment is much more curicial compared to other type
    of networks out there.
  - I was introduced to new mlop tools, like wandb.ai to evaluate the models, although this was introduced, i did not
    look further it it as i will still reading allot of papers and trying to make some validated notes to write on how
    labeling works.

- Within the first three days, i understood the core concepts that i was missing and applied them within my mathematical
notes.

- The next few days, was based on how labeling works within gans, which was the next step on moving forward; There are
different ways to label a model, and some caes the labels are very poor with great images, forcing one to actively
create the labels through some manual intervention, in this instance, as stated in the project plan, i would like to use
a deep neural network to do so, before the images get parsed into the GAN. To do this, what i did was the following
  - Kaggle : Looking at examples of how different datasets have been pre processed, based on their labels, or quality
    of the image.
  - Also looked at different examples of datasets
    - cats and dogs
    - Humans
    - Anime faces | three varients, although i will stick with the main dataset as its very nicely labeled. Further
            labeling will allow for better reproduction when creating an image .
- I also looked into papers regarding WGAN, and CGAN, mainly because they presented cool ideas

# Observations

While my understanding on the maths had greatly improved i noticed that there were still issues regarding or trying to
understand how labeling works, with respect to the ggenerative network, within this week, i founded that i needed to
read more into conditional gans, move over, CDCGANs, conditional Deep convolutional generative adverserial networks.

With that being said, i also noticed that there were minimal amount of papers regarding active labeling using DNN. In
this instance i decided to go out my way to contact a friend of mine who is a researcher at kings college london. To
provide me better insights on how labeling may work through NN.

# Self Observation and What Next

With that being said, this week was quite productive, and have a list of papers to now red more into, for the following
week. Furthermore I will start working on the project, and work more on loading the dataset into the project it self.
