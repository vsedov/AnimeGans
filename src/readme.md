# Evaluation of Illustration2Vec [ Core / default model ]

This README outlines the steps to evaluate the Illustration2Vec model.

## Prerequisites

    - The dataset should already be loaded and you should be using the core/default database.
    - You should have cd to the project directory: project/.

## Steps to evaluate the model

    - cd to the illustration2vec directory: project/src/create_data/illustration2vec/.
    - Run the script get_models.sh to obtain the necessary models.
    - Run the script gen_tags.py to generate tags for the illustrations.
    - cd to the parent directory: project/src/create_data/.
    - Run the script create_csv.py to create a CSV file from the generated tags.
    - cd to the project root: project/.
    - Run the following command to train the model with desired parameters:

After following these steps, the Illustration2Vec model should be ready for evaluation. The generated tags and CSV files should be in the correct format for use with the model.
