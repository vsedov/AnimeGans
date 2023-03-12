
# Feat

- Additional reading about WGANS and its operation
- Completed preliminary processing of both the cats and dogs datasets
Train three distinct models for these two datasets, cats, dogs, and humans.
- Training for each dataset requires a considerable amount of time, during which I will concentrate on my senior thesis.
This week, wandb.qai was completely built, providing me with a highly dynamic approach to test and refer to gradients. This is one of the most important details relating this procedure.

This was also the week where where I saw saw some of the core issues that I had learnt in weeks 1 and 2 in action, where I saw a live view of mode collapse, although as stated before there wasnt much quntantitative measure regarding how to measure mode collapse, FID distances and more are great but do not tell to a high degree, so I had to review the gradients if mode collapse did occur, and how the model would avoid mode collapse as well, these observations are all detailed

# Observations

- WGAN is different from traditional generative adversarial networks (GANs) in several ways. The most significant difference is that WGANs use the Wasserstein distance, also known as the Earth Mover's distance, as the loss function for the discriminator, instead of the traditional binary cross-entropy loss. This allows the discriminator to learn a smoother and more accurate estimate of the true data distribution, which in turn allows the generator to produce more realistic samples.

- Another significant difference is that WGANs use a gradient penalty term in the loss function for the discriminator, which encourages the discriminator to be Lipschitz continuous and prevents it from collapsing to a degenerate solution. This helps to stabilize the training of the GAN and improves its performance.

- Additionally, WGANs impose a weight clipping constraint on the discriminator, which prevents it from becoming too powerful and allows the generator to learn a more balanced distribution of the data. This is different from traditional GANs, which do not have such a constraint and can suffer from issues such as mode collapse.

# Self Observation and What Next

So whilst I was training the cats and dogs dataset, I will write my report and mathematical concepts for the intermediate report. I am also monitoring and charting the learning process in real-time using wandb.ai.

I also intended to test wgans, but I did not have enough time to properly analyse them, so I will review them in the future with a limited number of epochs to see how they compare to other models, however it is crucial to remember that this is insufficient and would just offer a high-level perspective.

The training for the cats and dogs dataset was divided into cats and dogs; I did not combine the labels since I believed it would be pretty unusual to have a strange blend of cats and dogs.
Not only would this further confound the model, but the datasets themselves are also poorly constructed.
