device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
import os
import zipfile
from pathlib import Path
import requests
def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    data_path = Path("data/")
    image_path = data_path / destination
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)        
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)
        if remove_source:
            os.remove(data_path / target_file)    
    return image_path
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
manual_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])           
print(f"Manually created transforms: {manual_transforms}")
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=manual_transforms, batch_size=32)
train_dir = image_path / "train"
test_dir = image_path / "test"
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
automatic_transforms = weights.transforms() 
print(f"Automatically created transforms: {automatic_transforms}")
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders( train_dir=train_dir, test_dir=test_dir, transform=automatic_transforms, batch_size=32)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
for param in model.features.parameters():
    param.requires_grad = False
set_seeds() 
model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features=1280, out_features=len(class_names), bias=True).to(device))
from torchinfo import summary
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from typing import Dict, List
from tqdm.auto import tqdm
from going_modular.going_modular.engine import train_step, test_step
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
          epochs: int, device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [] }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn,  optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn,  device=device)
        print(f"Epoch: {epoch+1} | " f"train_loss: {train_loss:.4f} | "  f"train_acc: {train_acc:.4f} | "  f"test_loss: {test_loss:.4f} | "  f"test_acc: {test_acc:.4f}"  )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc}, global_step=epoch)        
        writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))    
    writer.close()
    return results
set_seeds()
results = train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=5, device=device)

def create_writer(experiment_name: str, model_name: str, extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

example_writer = create_writer(experiment_name="data_10_percent", model_name="effnetb0", extra="5_epochs")
from typing import Dict, List
from tqdm.auto import tqdm
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,loss_fn: torch.nn.Module,
          epochs: int,device: torch.device, writer: torch.utils.tensorboard.writer.SummaryWriter ) -> Dict[str, List]:   
    results = {"train_loss": [],"train_acc": [],"test_loss": [],"test_acc": [] }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn,optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model,dataloader=test_dataloader, loss_fn=loss_fn,device=device)
        print( f"Epoch: {epoch+1} | " f"train_loss: {train_loss:.4f} | " f"train_acc: {train_acc:.4f} | " f"test_loss: {test_loss:.4f} | " f"test_acc: {test_acc:.4f}"  )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if writer:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc}, global_step=epoch)
            writer.close()
        else:
            pass
    return results
data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",  destination="pizza_steak_sushi_20_percent")
train_dir_10_percent = data_10_percent_path / "train"
train_dir_20_percent = data_20_percent_path / "train"
test_dir = data_10_percent_path / "test"
print(f"Training directory 10%: {train_dir_10_percent}")
print(f"Training directory 20%: {train_dir_20_percent}")
print(f"Testing directory: {test_dir}")
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
BATCH_SIZE = 32
train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_10_percent,test_dir=test_dir, transform=simple_transform, batch_size=BATCH_SIZE)
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent, test_dir=test_dir, transform=simple_transform, batch_size=BATCH_SIZE)
print(f"Number of batches of size {BATCH_SIZE} in 10 percent training data: {len(train_dataloader_10_percent)}")
print(f"Number of batches of size {BATCH_SIZE} in 20 percent training data: {len(train_dataloader_20_percent)}")
print(f"Number of batches of size {BATCH_SIZE} in testing data: {len(train_dataloader_10_percent)} (all experiments will use the same test set)")
print(f"Number of classes: {len(class_names)}, class names: {class_names}")
import torchvision
from torchinfo import summary
effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)
print(f"Number of in_features to final layer of EfficientNetB2: {len(effnetb2.classifier.state_dict()['1.weight'][0])}")
import torchvision
from torch import nn

OUT_FEATURES = len(class_names)
def create_effnetb0():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    set_seeds()
    model.classifier = nn.Sequential( nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=OUT_FEATURES) ).to(device)
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model
def create_effnetb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    set_seeds()
    model.classifier = nn.Sequential( nn.Dropout(p=0.3),nn.Linear(in_features=1408, out_features=OUT_FEATURES)).to(device)
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model
effnetb0 = create_effnetb0() 
effnetb2 = create_effnetb2()
num_epochs = [5, 10]
models = ["effnetb0", "effnetb2"]
train_dataloaders = {"data_10_percent": train_dataloader_10_percent, "data_20_percent": train_dataloader_20_percent}

from going_modular.going_modular.utils import save_model

set_seeds(seed=42)
experiment_number = 0
for dataloader_name, train_dataloader in train_dataloaders.items():
    for epochs in num_epochs: 
        for model_name in models:
            experiment_number += 1
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] DataLoader: {dataloader_name}")
            print(f"[INFO] Number of epochs: {epochs}")  
            if model_name == "effnetb0":
                model = create_effnetb0() # creates a new model each time (important because we want each experiment to start from scratch)
            else:
                model = create_effnetb2() # creates a new model each time (important because we want each experiment to start from scratch)            
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,optimizer=optimizer,loss_fn=loss_fn, epochs=epochs, device=device, 
                  writer=create_writer(experiment_name=dataloader_name, model_name=model_name, extra=f"{epochs}_epochs"))            
            save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
            save_model(model=model, target_dir="models", model_name=save_filepath)
            print("-"*50 + "\n")

best_model_path = "models/07_effnetb2_data_20_percent_10_epochs.pth"
best_model = create_effnetb2()
best_model.load_state_dict(torch.load(best_model_path))
from pathlib import Path
effnetb2_model_size = Path(best_model_path).stat().st_size // (1024*1024)
print(f"EfficientNetB2 feature extractor model size: {effnetb2_model_size} MB")
from going_modular.going_modular.predictions import pred_and_plot_image
import random
num_images_to_plot = 3
test_image_path_list = list(Path(data_20_percent_path / "test").glob("*/*.jpg")) 
test_image_path_sample = random.sample(population=test_image_path_list, k=num_images_to_plot)
for image_path in test_image_path_sample:
    pred_and_plot_image(model=best_model,image_path=image_path,class_names=class_names, image_size=(224, 224))
import requests
custom_image_path = Path("data/04-pizza-dad.jpeg")
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")
pred_and_plot_image(model=model, image_path=custom_image_path, class_names=class_names)















