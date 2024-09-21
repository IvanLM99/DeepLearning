import os
import zipfile
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datetime import datetime
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_dataset(zip_file_path, remove_zip=True):

    # Check if the dataset folder already exists
    dataset_folder = os.path.splitext(zip_file_path)[0]
    if os.path.exists(dataset_folder):
        print(f"Dataset folder already exists: {dataset_folder}")
        return dataset_folder

    # Check if the zip file exists
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError("Zip file not found.")

    # Create directory to extract the files
    os.makedirs(dataset_folder, exist_ok=True)

    # Extract the dataset file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print('Extracting files...')
        zip_ref.extractall(os.path.dirname(zip_file_path))
        print('Extraction finished.')

    # Remove the zip file
    if remove_zip:
        os.remove(zip_file_path)
        print('Zip file removed.')

    # Return the path to the extracted dataset
    return dataset_folder


def show(tensor, num=25):
    """
    A function that visualizes a tensor as an image grid.

    Args:
        tensor (tensor): Batch of images
        num (int, optional): Number of images to show. Defaults to 25.
    """

    # Clip the value of number of images (num)
    num = max(25, min(num, len(tensor)))

    # Detach tensor, move it to the gpu and scale
    data = (tensor.detach().cpu() + 1) * 0.5

    # Make grid of 5x5
    grid = make_grid(data[num-25:num], nrow=5).permute(1, 2, 0)

    # Plot the figure
    plt.figure(figsize=(7, 7))
    plt.imshow(grid.clip(0, 1))
    plt.show()


def save_checkpoint(name, epoch, config, path):
    """
    Save checkpoint (the Generator and the Critic)
    """

    # Get the current time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the models
    torch.save({'epoch': epoch,
                'timestamp': timestamp,
                'gener_model_state_dict': config['generator'].state_dict(),
                'g_optimizer_state_dict': config['g_optimizer'].state_dict(),
                'discr_model_state_dict': config['discriminator'].state_dict(),
                'd_optimizer_state_dict': config['d_optimizer'].state_dict()},
               os.path.join(path, f"GAN-{name}.pkl"))


def load_checkpoint(name, config, path):
    """
    Load checkpoint (the Generator and the Critic)
    """
    # Define the file path
    file_path = os.path.join(path, f"GAN-{name}.pkl")

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Checkpoint file {file_path} not found.")
        return None

    # Load the checkpoint
    checkpoint = torch.load(file_path)

    # Load model and optimizer states
    config['generator'].load_state_dict(checkpoint['gener_model_state_dict'])
    config['g_optimizer'].load_state_dict(checkpoint['g_optimizer_state_dict'])
    config['discriminator'].load_state_dict(checkpoint['discr_model_state_dict'])
    config['d_optimizer'].load_state_dict(checkpoint['d_optimizer_state_dict'])

    # Load epoch
    config['epoch'] = checkpoint['epoch']

    # Print the time of the checkpoint
    print(f"Checkpoint was saved on: {checkpoint.get('timestamp', 'Timestamp not found')}")

    # Optionally, return loaded objects for further processing
    return {
        'epoch': checkpoint['epoch'],
        'generator': config['generator'],
        'g_optimizer': config['g_optimizer'],
        'discriminator': config['discriminator'],
        'd_optimizer': config['d_optimizer']}


def visual_epoch(fake_imgs, real_imgs, gener_losses_epoch_list,
                 discr_losses_epoch_list):
    """
    A function to display images and training at the end of an epoch.

    It shows real images, fake images and the training losses of generator and
    discriminator.
    """
    plt.close('all')
    show(fake_imgs)
    show(real_imgs)
    plt.figure(figsize=(10, 5))
    plt.plot(gener_losses_epoch_list, label="Generator Loss")
    plt.plot(discr_losses_epoch_list, label="Critic Loss")
    plt.legend()
    plt.show()

