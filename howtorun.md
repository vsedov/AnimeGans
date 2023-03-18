# How to Run

This project is an implementation of the Illustration2Vec model, which is a deep
neural network for tagging and predicting attributes of illustrations. Here are
the steps to run the project: With ACGANS

1. Poetry init / poetry install
2. `cd src/create_data/`  and run `python generate_data_images.py`
3. `cd illustrationtovec` and run `get_models.sh`
4. run `python get_tags.py`
    Note: This step may take several hours, depending on the size of your dataset.
5. Now `cd ..` and run `python create_csv.py`

Now that you have the data and the images preproc, you May now run the main file
or you May wish to run the other possible options in train / test directly.
