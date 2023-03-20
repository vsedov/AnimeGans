# Train.py Usage Guide

The train.py script is used to train a deep learning model. The script has several command line arguments that can be specified to customize the training process.

## Changing tags

As of the moment I have not been able to implement this within the gui or the
argparse as this system is very unstable.

## Command Line Arguments

Here is a list of all the available command line arguments for the train.py script:

```python
hair = [
    "orange",
    "white",  #
    "aqua",
    "gray",  #
    "green",
    "red",
    "purple",
    "pink",  #
    "blue",
    "black",
    "brown",  #
    "blonde",
]
eyes = [
    "gray",
    "black",
    "orange",
    "pink",
    "yellow",
    "aqua",
    "purple",
    "green",
    "brown",
    "red",
    "blue",
]
```

You can comment out any of the above to see if you can be specific with respect
to your core generation, but it is advised to keep a note of this somewhere just
in case. As generating images from this May be a bit hard.
Reason for implementation : Atm even with the larger dataset. the tag system/labeling
system is very off, such that I hope to improve it by reducing the labels as
referenced in the ACGAN Paper.

```bash

-h, --help            show this help message and exit
-i ITERATIONS, --iterations ITERATIONS
                      Number of iterations to train Generator
-G EXTRA_GENERATOR_LAYERS, --extra_generator_layers EXTRA_GENERATOR_LAYERS
                      Number of extra layers to train Generator
-D EXTRA_DISCRIMINATOR_LAYERS, --extra_discriminator_layers EXTRA_DISCRIMINATOR_LAYERS
                      Number of extra layers to train Discriminator
-C CHECK_POINT_SAVE_SPLIT , --check_point_save_split  CHECK_POINT_SAVE_SPLIT
                      Add a checkpoint split, Number of epochs you want to save your models
-b BATCH_SIZE, --batch_size BATCH_SIZE
                      Training batch size
-s SAMPLE_DIR, --sample_dir SAMPLE_DIR
                      Directory to store generated images
-c CHECKPOINT_DIR, --checkpoint_dir CHECKPOINT_DIR
                      Directory to save model checkpoints
--sample SAMPLE       Sample every n steps
--lr LR               Learning rate gen and discriminator
--beta BETA           Momentum term in Adam optimizer
--wandb WANDB         Use wandb
--wandb_project WANDB_PROJECT
                      Use Project_scope
--wandb_name WANDB_NAME
                      Use project_name
--overwrite OVERWRITE
                      Path overwrite, such that if you wish to use this : Batchsize:epoch_ammount is required for given directory 64:120
-t EXTRA_TRAIN_MODEL_TYPE, --extra_train_model_type EXTRA_TRAIN_MODEL_TYPE
                      Use best model instead [number:{}, best]
```

## Example Usage

```bash
python train.py -i 250 -b 128 -G 1 -D 0  --wandb false
```

In this example, the script will train the model for 250 epochs with a batch size of 128 and an extra layer in the generator. The wandb argument is set to false.

If you want to continue training from a previous run, you can specify the overwrite argument along with the previous run's batch size and epoch amount:

```bash
python train.py -i 10000 -b 128 -G 1 -D 0  --overwrite 128:250 --wandb true -t number:249


```

In this example, the script will continue training for an additional 10,000 epochs with the same batch size and generator layer as the previous run. The overwrite argument specifies the previous run's batch size (128) and epoch amount (250) and the -t argument specifies to use the highest numbered model.
