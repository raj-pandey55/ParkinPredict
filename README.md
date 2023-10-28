# ParkinPredict

This program implements a Parkinson's disease detection system using machine learning. The system is trained on a dataset of Parkinson's patients and healthy individuals, and uses a Support Vector Machine (SVM) classifier to predict whether a new individual has Parkinson's disease.

To use the system, simply provide it with a list of the individual's features, and it will return a prediction of whether or not the individual has Parkinson's disease.

# Features -:

• name: The name of the individual.
• status: Whether the individual has Parkinson's disease (1) or is healthy (0).
• MDVP:Fo(Hz): The average vocal fundamental frequency.
• MDVP:Fhi(Hz): The maximum vocal fundamental frequency.
• MDVP:Flo(Hz): The minimum vocal fundamental frequency.
• MDVP:Jitter(%): The percentage of variation in vocal fundamental frequency.
• MDVP:Shimmer(%): The percentage of amplitude variation in vocal fundamental frequency.
• MDVP:APQ(Hz): The averaged perturbation quotient.
• Shimmer(dB): The decibel value of the shimmer signal.
• HNR: The harmonic-to-noise ratio.
• NHR: The noise-to-harmonics ratio.
• RPDE: The revised pitch deviation estimate.
• DFA: The detrended fluctuation analysis of the vocal fundamental frequency.
• PPE: The perturbation percentage estimate.

# Usage:

The dataset can be used to train a machine learning model to detect Parkinson's disease. The model can then be used to predict whether a new individual has Parkinson's disease based on their features.

• Install the required Python libraries:

• Download the Parkinson's Disease Detection Dataset:

• Run the following command to train the model:
Python
python model.py

• Once the model is trained, you can use it to make predictions on new data:

# Example:

Prediction: The Person has Parkinsons

This code achieves an accuracy of 88.8% on the test set.

# Notes:

• The system is trained on a relatively small dataset, so it is important to use it cautiously.
• The system is only as good as the data it is trained on. If the data is biased, the system will also be biased.
• The system is not a substitute for medical advice. If you are concerned about having Parkinson's disease, please consult a doctor.
