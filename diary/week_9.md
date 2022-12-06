
# Feat

- So within this week, I had trained multiple variants of the cats and dogs dataset, ranging from 30 to 1k eopchs. The cats and dogs dataset is quite low, has bad data, and although nicely labelled, is very poorly specified, which means that the results that were outputted are quite poor. However, this is a good thing, as it demonstrates that a validation method is required to ensure the quality of the method and results.
Thus, when I deal with the core dataset, I will have a few validation techniques to verify that the dataste and photos being processed are of the highest quality.

-

This week, I spent around five days training the human dataset. This was an extremely intense effort that required my computer to be on for an extended amount of time. At the time, I was unaware that there were school servers that I could have utilised to train my network.

- Therefore, not only will I be training the hyman dataset, but I have also logged a list of mode collapses that occurred with the prior dataset and updated the model accordingly; the alterations should be included in the report.

-

I've also sought additional advice regarding mode collapse and how to evaluate it; at the time of writing, I had stated that there was no mathematically viable way to measure mode collapse; however, the theory of measuring mode collapse through cosine waves appears to be a viable option; this is to be tested, and I'm unsure if it will work.

# Observations

- Wandb evaluated any issues with vanishing gradient well; however, the hyuman dataset had vanishing gradient issues for the first 30 epochs, which really concerned me. Additionally, the mode l was having trouble learning from the initial modifications made to avoid mode collapse, which was, to say the least, an odd observation.
However, after 30 epochs it figured out how the pictures appeared, and the resultant output was correct.

This piqued my curiosity to the point that I paused my training to determine why or what the situation was with reference to this problem.
As a result, I decided to add a 2-momentum ratio  between the generator and the discriminator in order to increase the learning rate.
This resolved the problem

# Self Observation and What Next

By the conclusion of this week, I had acquired a great deal of fresh, intriguing notions.
Although some may not be presented in the report due to relevance, moving forward I will redo my entire report in a research-based manner, and restructure all my findings correctly. I will also reevaluate my research in a more mathematical manner, without having a very messy layout regarding where to place data, which I have discovered happens quite often with me.
