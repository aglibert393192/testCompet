Fine-tune a convolutional neural network to classify X-ray images into two categories: NORMAL and PNEUMONIA.

- Load the pre-trained ResNet-18 model in a variable called `resnet18`.

- Perform the appropriate adjustments to the model such that **only the weights in the last layer** are updated during training. Save any adjustments you make to the final layer in a variable `resnet18.fc`.

- Fine-tune your adjusted ResNet-18 model for ** only 3 epochs ** using the data from `train_loader`. You don't have to use a validation set during fine-tuning. Run the provided validation cell to see your model performance; print the `test_accuracy` and `test_f1_score` and round to three decimal points. 

 ---

**BONUS**: After the submission, you can play with your model further to see how well it can be trained. You can train it for more epochs and with a validation set. At the end of the provided solution, you can find a sample code that you can run to divide the training data into `train` and `val` subsets.