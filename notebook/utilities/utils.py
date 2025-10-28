import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set(style="whitegrid")


def _add_textbox_at_last_datapoint(text, y_values, fontsize=12):
    last_y_value = y_values[-1]
    plt.text(
        len(y_values) + 0.1,
        last_y_value, 
        text, 
        fontsize=fontsize,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )


def plot_dataloading_results(logfile_path, fontsize=12, figsize=(10, 6), plot_for='mp'):
    """Plots epoch times from log data with and without caching for comparison."""

    # Initialize plot with the specified figure size
    plt.figure(figsize=figsize)

    # Load JSON data into a DataFrame and transpose it for easier access
    df = pd.read_json(logfile_path, orient='index').T
    
    # Extract columns representing epoch times
    epoch_columns = df.filter(regex=r'^t_epoch_')
    epoch_numbers = [int(col.split('_')[-1]) for col in epoch_columns.columns]

    # Calculate extrapolated times for epochs without caching
    cold_epoch_time = epoch_columns.iloc[0, 0]  # First epoch time as baseline
    epoch_times_without_cache = [cold_epoch_time] * len(epoch_numbers)  # Repeat for all epochs

    # Select right descriptions based on benchmark type
    if plot_for == 'mp':
        label_line_1 = "Actual epoch times WITH caching"
        label_line_2 = "Extrapolated epoch times WITHOUT caching"
        label_subtitle = "using Mountpoint for S3 with and without caching"
    elif plot_for == 's3pt':
        label_line_1 = "Actual epoch times when streaming dataset on each epoch"
        label_line_2 = None
        label_subtitle = "using S3 Connector for PyTorch"
        
    
    # Plot actual epoch times and extrapolated times
    plt.plot(
        epoch_numbers, epoch_columns.values[0], 'ro-', 
        linewidth=2, markersize=10, 
        label=label_line_1
    )

    if label_line_2:
        plt.plot(
            epoch_numbers, epoch_times_without_cache, 'bD--', 
            linewidth=2, 
            label=label_line_2
        )
        
    # Configure plot labels, title, and styling
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title(f"Time per Epoch \n ({label_subtitle}) \n", fontsize=fontsize + 4, fontweight="bold")
    plt.xticks(epoch_numbers, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Add estimated total training time as a textbox at the last data points
    textbox_template = "Total training time: ~{:.0f}s"
    _add_textbox_at_last_datapoint(
        textbox_template.format(df['training_time'].iloc[0]), epoch_columns.values[0]
    )
    
    if label_line_2:
        _add_textbox_at_last_datapoint(
            textbox_template.format(cold_epoch_time * len(epoch_numbers)), epoch_times_without_cache
        )
    
    # Scale y-axis based on max epoch time with caching
    plt.ylim(0, max(epoch_columns.values[0]) * 1.5)
    plt.legend(fontsize=fontsize)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def plot_checkpointing_results(logfile_dict, fontsize=12, figsize=(10, 6)):
    """Plots checkpoint times from log files and highlights averages."""

    # Initialize the plot with the specified figure size
    plt.figure(figsize=figsize)
    plt_stls = ['ro-', 'bD-', 'gv', 'm<', 'b>', 'c^']  # Styles for plot lines
    max_ckpt_time = 0  # Track the max checkpoint time across all logs

    def _read_values(json_path):
        """Reads and extracts checkpoint times from a JSON log file."""
        df = pd.read_json(json_path, orient='index').T
        ckpt_columns = df.filter(regex=r'^t_ckpt_\d+$')
        ckpt_times = ckpt_columns.values[0]
        return ckpt_times

    # Process and plot each log file in the dictionary
    for i, (ckpt_label, ckpt_logfile) in enumerate(logfile_dict.items()):
        ckpt_times = _read_values(ckpt_logfile)
        ckpt_numbers = list(range(1, len(ckpt_times) + 1))
        
        # Plot individual checkpoint times
        plt.plot(
            ckpt_numbers, ckpt_times, 
            plt_stls[i] if i < len(plt_stls) else None,
            linewidth=2, 
            label=f"Individual checkpoint times to {ckpt_label}"
        )
        
        # Plot average checkpoint time as a dashed line
        plt.axhline(
            y=sum(ckpt_times) / len(ckpt_times), 
            color=plt_stls[i][0] if i < len(plt_stls) else None,
            linestyle='--', 
            label=f"Average checkpoint time to {ckpt_label}"
        )

        # Update max checkpoint time if the current max is higher
        max_ckpt_time = max(max_ckpt_time, max(ckpt_times))
    
    # Configure plot labels, title, and styling
    plt.xlabel("Checkpoint Number")
    plt.ylabel("Time (seconds)")
    plt.title(f"Model Checkpointing", fontsize=fontsize + 4, fontweight="bold")
    plt.xticks(ckpt_numbers, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.ylim(0, max_ckpt_time * 1.5)  # Scale y-axis based on max checkpoint time
    plt.legend(fontsize=fontsize)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def visualize_images(dataset_path, num_images_per_class=5, image_size=(128, 128), verbose=False):
    """
    Visualizes N images per class in a KxN grid from a dataset organized in subfolders by class.
    
    Parameters:
        dataset_path (str): Path to the dataset where each class has its own subfolder.
        num_images_per_class (int): Number of images to sample and display per class.
        image_size (tuple): Desired image size for visualization (width, height).
        verbose (bool): Flag to enable or disable timing printouts.
    """
    start_time = time.time()
    
    # Step 1: List subdirectories (each one representing a class)
    start_list_classes = time.time()
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(classes)
    if verbose:
        print(f"Listing classes took {time.time() - start_list_classes:.2f} seconds")
    
    # Create a plot grid with K rows (classes) and N columns (images per class)
    fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(num_images_per_class * 2, num_classes * 2))
    fig.suptitle(f'{num_classes} Classes - {num_images_per_class} Images per Class', fontsize=16)

    # Step 2: Load images for each class and label the rows
    for i, class_name in enumerate(classes):

        label = f"Class '{class_name}'"
        class_path = os.path.join(dataset_path, class_name)
        
        # Optimize the image selection process
        start_images_listing = time.time()
        # List all images in the class directory
        all_images = [img for img in os.listdir(class_path) if img.endswith(".jpg")]
        
        # Randomly select images
        if len(all_images) < num_images_per_class:
            selected_images = all_images  # Select all if fewer images are available
        else:
            selected_images = random.sample(all_images, num_images_per_class)
        
        if verbose:
            print(f"Selecting images for class '{class_name}' took {time.time() - start_images_listing:.2f} seconds")

        # Display label centered in front of each row
        y_position = 1 - ((i + 0.6) / num_classes)
        fig.text(0.06, y_position, label, ha='center', va='center', rotation=90, fontsize=12)

        # Load and display images
        for j, image_name in enumerate(selected_images):
            image_path = os.path.join(class_path, image_name)
            ax = axes[i, j] if num_classes > 1 else axes[j]
            
            # Load and resize image
            img = Image.open(image_path).resize(image_size)
            img_array = np.array(img)  # Convert to NumPy array for pixel value calculation
            
            # Calculate average pixel value
            avg_pixel_value = np.mean(img_array)
            
            # Display image
            ax.imshow(img)
            ax.axis('off')  # Hide axes
            
            # Display average pixel value in upper right corner
            ax.text(0.95, 0.95, f'Avg: {avg_pixel_value:.2f}', ha='right', va='top', 
                    transform=ax.transAxes, color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Final adjustment and show plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.14)  # Adjust left margin for centered row labels
    if verbose:
        print(f"Total visualization time: {time.time() - start_time:.2f} seconds")
    plt.show()


def wait_for_job_to_finish(job_id, client, check_interval=10):
    
    while True:
        job_info = client.get_job_info(job_id)
        status = job_info.status
        
        if status == 'SUCCEEDED':
            print(f"Job '{job_id}' has finished.")
            return
        elif status == 'FAILED':
            raise Exception(f"Job '{job_id}' has failed.")
        elif status == 'STOPPED':
            raise Exception(f"Job '{job_id}' has been stopped.")
        else:
            time.sleep(check_interval)
