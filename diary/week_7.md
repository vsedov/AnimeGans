# Feat

- Start the proper production code for the initial idea on how DCGANS work
In this portion of the first stint of the week i created the initial section for the generative network, and i had also test the model, using wandb ai, tis helped formally evaluate my model, before applying any dataset.

- In sync with creating the model, I also started to create the initial dataset loader fclasses for the chats and dogs datasets, when i fetch data from kaggle. Meaning i created a few helper functions for both kaggle and wandb.ai [ Once again this was introduced from a Senior machine learning engineer and was highly recommend ]

- Loading the dataset was harder than I thought so working on how to split the labels was a priority before i started any training.

# Observations

Not many observations were made here, the data splitting was mainly done through a helper class that i had made that
uses torch to split the data and im using a run time constant class, that allows me to dynamicly change the constants or
change the values when ever required, instead of loading a file or something i can update them through a class, this is
class injection, a new concept that i had learnt from one of my previous machine learning projects.

Regarding torch, looking more into the documentation, i noticed that torch can do allot here is just a few things that
I tried messing around with

- Data can be split into smaller parts for training and validation.
- Models can be trained on both the training data and the validation data.
- Models can be evaluated on the validation data to measure accuracy and compare different models.
- Models can be saved and reused for other tasks.
- Large datasets can be split into batches of data to make training more efficient.
- Models can be fine-tuned on the validation set.
- Data can be augmented and preprocessed before being split into training and validation sets.
- Data can be split into multiple subsets for cross-validation.
- Data can be split into training and validation sets using stratified sampling or other methods.
- Data can be split using random sampling or based on specific attributes.

# Self Observation and What Next

This was another quite productive week, although very stressful, due to the Multi agents coursework, hence development
slowed down once again. In the next few weeks i want to quickly test the models for any errors, in the code base and
start the development and integration with wandb.ai with the the entire scope of the project.
