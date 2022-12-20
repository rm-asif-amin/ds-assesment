# problem 2 

# Image Classifier
A deep neural net based classifier to classify images across 4 categories. The unique class names are- berry, bird, dog, flower. 

## Model Performance
Model performance on the provided test set -

```precision    recall  f1-score   support

       berry       0.94      0.86      0.90       400
        bird       0.91      0.96      0.94       400
         dog       0.94      0.95      0.95       400
      flower       0.94      0.95      0.95       400

    accuracy                           0.93      1600
   macro avg       0.93      0.93      0.93      1600
weighted avg       0.93      0.93      0.93      1600
```

## Usage

There's two ways to run this implementation-

1. Through a Docker Container after building Image from provided Dockerfile.

Commands-
```bash
git clone https://github.com/rm-asif-amin/name-classifier-binary.git
cd ds-assesment/problem_2
docker run -it -v [Path/to/train/test/data]:/opt/applications/image_classifier/dataset --memory 4gb demo-classifier:latest -epochs 25
```
epochs can be a smaller number for quick demonstration but test results will be poorer.
[Path/to/train/test/data] must be be absolute path to the folder containing train and test data.

2. Directly through python script train.py
Use from the command line with the following mandatory arguments '--dataset-path <path-to-data>' and '--saved-model-folder <path-to-model-destination>'. 
--dataset-path must contain two folders:
 - train
 - test
*both paths should be absolute.*

Commands-
```bash
git clone https://github.com/rm-asif-amin/name-classifier-binary.git
cd ds-assesment/
pip3 install -r requirements.txt
cd problem_2
python3 train.py --dataset-path <path-to-data> and --saved-model-folder <path-to-model-destination>
```
  

> **Warning**
> Docker Image Doesn't support GPU acceleration.
> Using directly through python script(method 2) will use GPU if CUDA is enabled and provide ~10X faster training.
 


### Full list or command line arguments-


| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `dataset-path` | `string` | **Required**. Input Folder for Training Data|
| `saved-model-folder` | `string` | **Required**. Output Folder for Trained Model( saved as 'trained_model.pt') |
| `evaluate` | `Bool` | **optional**. Controls whether to evaluate on test set. |
| `epochs` | `int` | **optional**. Number of passes made over the whole training set. Use smaller number for debugging     |
| `lr` | `float` | **optional**. Learning Rate of optimizer                                |
