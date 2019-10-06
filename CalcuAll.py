from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import *
from PIL import Image
import helpers.ProcessImage
import torch
import torch.nn.functional as F
import torchvision 
import torchvision.models as models
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

def create_model(arch, hidden_units):
    '''
        Creates a pretrained model using alexnet and returns the model
        
        Inputs:
        arch - The architecture to be used.
        hidden_units - The number of units in the hidden layer
        
        Outputs:
        model - The created (loaded) pretrained model
    '''
    # Define a new, untrained feed-forward network as a classifier
    print("Creating the model...")

    # Load a pretrained model
    if arch.lower() == "alexnet":
        model = models.alexnet(pretrained=True)
        input_features = 9216
    elif arch.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
        input_features = 512
    else:
        # We dont support the entered model architecture so return to start over
        print("Model architecture: {}, is not supported.".format(arch.lower()))
        return 0
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model    
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.Linear(hidden_units, 102),
                              nn.LogSoftmax(dim=1))
    
    if arch.lower() == 'alexnet':
        model.classifier = classifier
    elif arch.lower() == 'resnet18':
        model.fc = classifier
    
    print("Done creating the model\n")
    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs):
    '''
        Trains a model using a given loss function, optimizer, dataloaders, epochs, and whether or not to use the GPU. Outputs loss and accuracy numbers
        
        Inputs:
        model - The model to train
        train_loaders - The data for the training
        valid_loaders - The data for the validation
        criterion - The loss function 
        optimizer - The optimizer
        epochs - The number of epochs to run the training for
        
        Outputs:
        Prints out the training and validation losses and accuracies
    '''
    # Track the loss and accuracy on the validation set to determine the best hyperparameters
    print("Training the model...\n")

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the model for training
    model.to(device)

    # Define starters
    #epochs = 2 
    print_every = 10
    steps = 0
    running_loss = 0 
    
    #Keep track of the losses and accuracies for training and validation
    #train_losses, validation_losses, training_accuracies, validation_accuracies = [], [], [], []

    #Create for loop to train network
    for e in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            steps += 1
            #transfer model to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #train classifier layer using front and back
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()

            if steps % print_every == 0:
                # Initialize variables
                test_loss = 0
                accuracy = 0
                # switch to eval mode
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)

                        batch_loss = criterion(output, labels)
                        test_loss += batch_loss.item()

                        p = torch.exp(output)
                        top_p, top_class = p.topk(1, dim=1)
                        #equals = top_class == labels.view(*top_class.shape)
                        equality = (labels.data == p.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()               

                print("Epoch: {}\n".format(e+1),
                      "Training Loss: {}\n".format(running_loss/len(train_loader)),
                      "Validation Loss: {}\n".format(test_loss/len(valid_loader)),
                      "Accuracy: {}\n".format(accuracy/len(valid_loader) * 100))

                running_loss = 0                    

                # Make sure train is on 
                model.train()

    print("\nDone training the model \n")



def save_model(model, train_data, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):
    '''
        Saves a model to a checkpoint file with the learning rate, batch size, epochs, loss function, optimizer, hidden units, and architecture used in training
        
        Inputs:
        model - The model to train
        train_data - The dataset for the training. This is used to get the classes to indexes
        learning_rate - The learning rate used for training
        hidden_units - The hidden layers unit size
        arch - The architecture used
        save_directory - The directory to save the pth file to
        
        Outputs:
        Saves the checkpoint.pth file to the given directory
    '''
    print("Saving the model...")

    # Save the train image dataset
    model.class_to_idx = train_data.class_to_idx  
    
    if arch.lower() == "alexnet":
        input_features = 9216
    elif arch.lower() == "resnet18":
        input_features = 512
        
    checkpoint = {'input_size': input_features,
            'output_size': 102,
            'hidden_units': hidden_units,
            'epochs': epochs,
            'arch': arch,
            'learning_rate': learning_rate,
            'classifier': model.classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, 'mycheckpoint.pth')
    print("Done saving the model")


def load_model(checkpoint_file):
    '''
        Loads a model using a checkpoint.pth file
        
        Inputs:
        checkpoint_file - The file path and name for the checkpoint
        
        Outputs:
        model - Returns the loaded model
    '''
    print("Loading the model...")
    #Load the model from dictionary made
    checkpoint = torch.load(checkpoint_file)
    
    if(checkpoint['arch'].lower() == 'alexnet' or checkpoint['arch'].lower() == 'resnet18'):
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_index = checkpoint['class_to_idx']
    #model.classifier.epochs = checkpoint['epochs']
    
    #Freeze to prevent backpropagation
    for param in model.parameters():
        param.requires_grad = False    

    print("Done loading the model")
    return model    

def predict(categories, image_path, model, topk=5):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
        
        Inputs:
        image_path - The path and file name to the image to predict
        use_gpu - Whether or not to use the gpu for inference
        topk - The top n restults of the inference
        
        Outputs:
        top_p - The probabilities for the predictions
        labels - The class labels for the predictions
    '''
    image = process_image(image_path)
    
    #Switch to evaluation mode
    model.eval()
    
    #Use GPU if intended
    model.to(device)
    
    image = torch.from_numpy(np.array([image])).float()
    
    with torch.no_grad():
        output = model.forward(image)
        probs, classes = torch.topk(output, topk)
        probs = probs.exp()
        
    #Get list of probs and classes
    top_p = probs.tolist()[0]
    top_cl = classes.tolist()[0]

    #Reverse the dict
    idx_to_class = {model.class_to_index[i]: i for i in model.class_to_index}
    # OR {val: key for key, val in model.class_to_idx.items()}
    # OR {v:k for k, v in model.class_to_idx.items()}

    #Get the correct indices
    labels = []
    for c in top_cl:
        labels.append(categories[idx_to_class[c]])

    return top_p, labels

def sanity_check(cat_to_name, file_path, model, index):
    '''
        A sanity check that shows the flower image that we are trying to infer and the outcome of the prediction
        
        Inputs:
        cat_to_name - The categories json file that maps the names of the flowers
        file_path - The path and file name to the image to predict
        model - The model to use for inference
        index - The index of the flower we are trying to infer
        
        Outputs:
        Prints the image of the flower we are trying to predict and a bar graph of the predicted labels and their probabilities
    '''

    fig = plot.figure(figsize = (6,10))

    #Create axes for flower
    ax = fig.add_axes([.2, .4, .445, .445])

    #Display + process image
    result = process_image('flowers/test/54/image_05402.jpg')
    ax = imshow(result, ax);

    #Title for graph
    index = 54
    ax.set_title(cat_to_name[str(index)])

    #Predict image
    probs, classes = predict('flowers/test/54/image_05402.jpg', model)

    #Displays bar graph with axes
    ax1 = fig.add_axes([0, -.355, .775, .775])

    # Get range for probabilities
    y_pos = np.arange(len(classes))

    # Plot as a horizontal bar graph
    plot.barh(y_pos, probs, align='center', alpha=0.5)
    plot.yticks(y_pos, classes)
    plot.xlabel('probabilities')
    plot.show()
