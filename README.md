# Final Year Project

# Comparitive study on how Gans work using Anime Face Data

Generating Facial images of Anime / Dogs and humans, via sketches and use of Deep Convolutional Gans
DCGAN.

Utilized PyTorch framework for development. Used a NVIDIA GeForce RTX Cards: GPU machine to facilitate training of the model.

Scope:
Datasets that are used:

- Cats and Dogs
- Actual Human Faces
- Anime Faces

<p align="center">
  <img
    src="https://raw.githubusercontent.com/catppuccin/catppuccin/dev/assets/footers/gray0_ctp_on_line.svg?sanitize=true"
  />
</p>

### Cats and Dogs

After around 300 epochs, the generator has learnt some degree of how a dog would look like, despite this not being realistic. As indicated in my initial project plan, this was something I wanted to avoid before beginning work on the anime face data.

With the first placeholder and concept to analyse how gans function, followed by a deeper understanding of the underlying concepts to develop the process; resulting in more plausible graphics in the future.

With Cats and Dogs, the learning had little success after 200 epochs, as seen by the included photos and mp4 movies in this project.
Even though the pictures were distinct and there was limited mode collapse, there was a small amount of overfitting and the discriminator class had an excessive advantage over the generator class.

These photos were likewise of various sizes and required proper compression.
In addition, it seems that more data processing will be necessary, since there are some really poor photos that must be eliminated for this dataset to be viable.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/catppuccin/catppuccin/dev/assets/footers/gray0_ctp_on_line.svg?sanitize=true"
  />
</p>

### Human Faces

While Cats and Dogs is a poorly structured dataset, Human Faces yielded beneficial results and was trained over 1000 epochs and 3 days.
Comparing the cosine distance of the gradients allowed me to get a deeper knowledge of how and why mode collapse occurs. As noted in the report, this does not offer a quantitative assessment of how severe the mode collapse is, but it does provide a clear picture of how and why it occurs.

In addition, at the conclusion of the training, I discovered that the convergence for human faces had failed owing to the gradient ascent change shown in the dcgan study.
To resolve this, I will need further time and network testing.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/catppuccin/catppuccin/dev/assets/footers/gray0_ctp_on_line.svg?sanitize=true"
  />
</p>

### Anime Faces *Core part of project*

<p align="center">
  <img
    src="https://raw.githubusercontent.com/catppuccin/catppuccin/dev/assets/footers/gray0_ctp_on_line.svg?sanitize=true"
  />
</p>

## How to run this project ?

### Constants

Constants are stored in

```
src/utils/constant.py
```

### Running Module

To run the module, and to train fake data, you can run python -m src
Although this will be changed in the future.

There are two core parameters at the time of writing that you will need to run; to test a model you  must go into the
constants file, and switch the boolean value | to run tests you must switch the run_test boolean operator

```
python -m src
```

### Future Scope

In the future, there will be a refactoring, on the 10th and will have a json file that you can reference to. Instead of
manually running data.

### Past

To view my initial gan and my structure please review the following commit
`3844e3ec`
This will contain the core code which allowed for this base line to occour.
