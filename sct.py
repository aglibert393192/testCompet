assert resnet18, "No ResNet-18 model has been found. Ensure your model is saved in a variable 'resnet18'"

for name, param in resnet18.named_parameters():
    if 'fc' in name:
        assert param.requires_grad == True, "The final layer of your ResNet-18 model should be trainable but is frozen."
    else:
        assert param.requires_grad == False, "All layers except the final layer of your ResNet-18 model should be frozen but are trainable."

assert resnet18.fc.in_features == 512, "The number of input features in the last layer of your network should match the original ResNet-18 architecture. You can check what's the appropriate number of input features by querying the `.in_features` property of the last ResNet-18 layer."

assert resnet18.fc.out_features == 1, "The number of output features should match the expected number of outputs in a standard binary classification task."

assert isinstance(test_accuracy, (float, torch.Tensor)), "The accuracy of your classifier is not a floating point number. Make sure to use the provided evaluation code."

assert test_accuracy >= 0.5, "The accuracy of your classifier seems too low. Aim for accuracy above 0.5. Consider changing your optimizer or its parameters. The Adam optimizer with learning rate of 0.01 is a good choice."

assert isinstance(test_f1_score, (float, torch.Tensor)), "The F1-score of your classifier is not a floating point number. Make sure to use the provided evaluation code."

assert test_f1_score > 0.1, "Your F1-score seems too low. Check your training loop and how you calculate the F1 metric. Consider changing your optimizer or its parameters. The Adam optimizer with learning rate of 0.01 is a good choice."


