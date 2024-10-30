"""
Trains a PyTorch image classification model using device agnostic code 
"""

import os 
import torch
import torchvision
from torchvision import transforms

import data_setup, engine, model_builder, utils, Get_data


# Setup the HYPERPARAMETERS 
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup the directories 
train_dir = Get_data.IMAGE_PATH / "train"
test_dir = Get_data.IMAGE_PATH / "test"

# Setup target device(device agnostic code)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms

train_transform_model = transforms.Compose([
                                      transforms.Resize(size=(64,64)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.TrivialAugmentWide(num_magnitude_bins=32),
                                      transforms.ToTensor()
                                   ])
test_transform_model = transforms.Compose([
                                    transforms.Resize(size=(64,64)),    
                                    transforms.ToTensor()
                                   ])

# Create DataLoaders with the help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               train_transform=train_transform_model,
                                                                               test_transform=test_model,
                                                                               batch_size=BATCH_SIZE,
                                                                               num_workers=os.cpu_count()
                                                                              )

# Create a model with the help of model_builder.py
Model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)
                             ).to(device)

# Set Loss and optimizer function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with the help of engine.py
engine.train(model=Model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_func=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device
            )

# Save the model with the help from utils.py
utils.save_model(model=Model,
                 target_dir="D:\PROJECT",
                 model_name="image_classfication_TinyVGG_model.pth")

