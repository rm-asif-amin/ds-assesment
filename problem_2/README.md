# problem 2 
Queries Answered in Discussions section. Plots added in Problem_2_figs.pdf to the repo.


# Image Classifier
A deep neural net based classifier to classify images across 4 categories. The unique class names are- berry, bird, dog, flower. 

## Model Performance
Model performance on the provided test set -

```
                 precision    recall  f1-score   support

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
docker build -t demo-classifier:latest .  
docker run -v [Path/to/train/test/data]:/opt/applications/image_classifier/dataset --memory=4g --shm-size=4g demo-classifier:latest -epochs 5
```
epochs can be a smaller number for quick demonstration but test results will be poorer.

[Path/to/train/test/data] must be be absolute path to the folder containing train and test data folders.

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
>Docker Image Doesn't support GPU acceleration right now.
>Using directly through python script(method 2) will use GPU if CUDA is enabled and provide ~10X faster training.
 


### Full list or command line arguments-


| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `dataset-path` | `string` | **Required**. Input Folder for Training Data|
| `saved-model-folder` | `string` | **Required**. Output Folder for Trained Model( saved as 'trained_model.pt') |
| `evaluate` | `Bool` | **optional**. Controls whether to evaluate on test set. |
| `epochs` | `int` | **optional**. Number of passes made over the whole training set. Use smaller number for debugging     |
| `lr` | `float` | **optional**. Learning Rate of optimizer                                |

## Discussions
We're using a pretrained resnet18 model from Pytorch's repository. Essentially we're loading a pretrained model and resetting the final fully connected layer with our own training data.
       
### Applying Augmentation-
We're applying a set of augmentations to the training and validation data to increase model's generalization ability-
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

However, since we apply an AutoAugment layer, each batch of images are augmented differently.Resulting augmented images can look like this-
       
![image](https://user-images.githubusercontent.com/50940878/208778896-2097de34-0807-41fe-a8ea-6037b73fb7b8.png)

       
### Plotting Training Metrics-
Training and Validation accuracy and loss plots are also available in Problem_2_figs.pdf 
       
![train_val_acc_step_7](https://user-images.githubusercontent.com/50940878/208777186-4a3c6a36-3463-408f-96fd-53e91c91718e.png)
![train_val_loss_step_7](https://user-images.githubusercontent.com/50940878/208777190-27ad002d-9ad1-4d50-b204-da092baf121f.png)

### Proving that model is not overfitted-
1. The first hint we get is from the very good performance on the training set. If the model had overfitted the training data, we probaly would see worse results for at least 1 category.
2. Secondly, as evident on the training metrics figures, the validation loss and accuracy closely follows the training curve. If the model had overfitted, it would perform noticably better on the training set.
3. Thirdly, Since out model was pretrained on Imagenet, it's weights are optimized for a large number of categories of images outside of the 4 we're re-training for. The model's inner layers are frozen and thus unable to learn anything from our training data and unable to overfit.
4. We're also applying random augmentation to the training images to prevent overfitting.
       
### Ensembling method-
1. Since the training process is set within the custom Model class, we can easily create models from different pre-trained models by changing a single paramater 
```python
derived_model = Model(pretrained_model=models.resnet18(pretrained=True),device=device)
```
2. Alongside the resnet18 model, we can train a vgg16 model by setting pretrained_model=models.vgg16(pretrained=True)
3. We can then implement a simple ensemble policy like maximum avg probability of the models' predictions
       
Wasn't implemented due to time constraints.
