# Observations of the following

# 10 - 10 - 400 epoch round

- After 200 steps of training with parameters -i 400 (400 iterations), an extra layer added to the generator, and a learning rate of 0.0001, the training produced decent outputs but failed after 200 iterations. The model was trained with an overwrite, starting with 10 iterations, followed by another 10, and finally with 400 iterations. Despite the promising results in the earlier stages, the training failed after 200 steps.
- There will be a test run of this going to 50

# Observations on Model Training

## Round 1

    - Trained a machine learning model with 400 iterations, an extra layer added to the generator, and a learning rate of 0.0001.
    - The training started with 10 iterations, followed by another 10, and finally with 400 iterations.
    - The model produced decent outputs but failed after 200 iterations despite showing promising results earlier.
    - A test run will be conducted with 50 iterations.

## Round 2

    - Trained a machine learning model with a batch size of 128 and 250 iterations.
    - The training provided a fruitful start compared to the previous round with a batch size of 64.
    - The training took longer than expected.
    - Mode collapse issue was not witnessed until epoch 499, where the model collapsed abruptly even though it was producing decent enough results to be valid.
    - The reason for the mode collapse issue was not clear. [Evaluating core
        principles weights and biases ]
