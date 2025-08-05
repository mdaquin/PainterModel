Model for predicting, from their bio, whether a painter has paintings in notable museums. 

# Data

The dataset is obtained from DBpedia, by querying it to obtain instances of painters, their description (abstract), the number of museums that can be found to have paintings by them, as well as additional information (categories, nationality, movement, country of birth when available). Since this is obtained from DBpedia, this data is under a CC-BY-SA 3.0 licence. ```data/prepare.py``` is used to balance and split the dataset, and saves the result in the ```data/train.csv``` and ```data/test.csv``` files. 

# Model 

The model is a simple text classifier created by adding a classification head on top of DistilBERT. ```train.py``` can be used to train the model over 10 epochs and will save the model having obtained the best accuracy on the validation set (```best_model.pth```).

# Activations

This model is made to test mechanistic interpretability methods, so the script ```activations.py``` can be used to create activation vectors for the layers of the model.

# Dependencies

This mostly relies on pandas, pytorch and huggingface transformers.