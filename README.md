# diabetic-retinopathy-detection

This repository contains a Python script for a Convolutional Neural Network (CNN) model to detect diabetic retinopathy from retinal images. The code uses TensorFlow and Keras for model development and evaluation.

### Workflow

1. **Data Preprocessing:**
   - Load and preprocess a dataset containing retinal images and corresponding labels.
   - Stratify the dataset into training, validation, and test sets.
   - Organize the data into working directories for each set.

2. **Model Architecture:**
   - Build a CNN model using TensorFlow and Keras.
   - Train the model on the training set and validate it on the validation set.

3. **Model Evaluation:**
   - Save the trained model.
   - Evaluate the model's performance on the test set.

4. **Diabetic Retinopathy Prediction:**
   - Use the trained model to predict the presence of diabetic retinopathy in a given retinal image.

### Usage

1. Install the required dependencies:

   pip install tensorflow matplotlib opencv-python pandas scikit-learn
   

3. Run the script:

   python your_script_name.py
   

4. Example usage for diabetic retinopathy prediction:
   
   from diabetic_retinopathy_detection import predict_class

   # Provide the path to the retinal image
   image_path = 'path/to/your/image.png'

   # Make a prediction
   predict_class(image_path)


### Notes

- Ensure that the dataset is properly formatted and available at the specified path.
- Adjust hyperparameters or model architecture as needed for your specific requirements.

### Acknowledgments

- This project uses TensorFlow and Keras for deep learning model development.
- The dataset used for training and evaluation is available at [link to dataset].
