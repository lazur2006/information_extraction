import matplotlib.pyplot as plt
from IPython.display import display

def update_training_plot(train_losses, val_metrics, epoch, total_epochs, plot_handle, metric_name="Macro-F1"):

    if not train_losses:
        return

    # Create figure
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()
    
    epochs_range = range(len(train_losses))
    
    # Plotting Training Loss
    lns1 = ax1.plot(epochs_range, train_losses, color='tab:red', 
                    label='Loss', marker='o', markersize=3, linewidth=1)
    
    # Plotting Validation Metric
    lns2 = ax2.plot(epochs_range, val_metrics, color='tab:blue', 
                    label=metric_name, marker='s', markersize=3, linewidth=1)
    
    # Formatting
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:red')
    ax2.set_ylabel(metric_name, color='tab:blue')
    
    # Fix scales for consistency
    ax2.set_ylim(0, max(1.0, max(val_metrics) if val_metrics else 1.0) * 1.1)
    
    plt.title(f"Training Progress: Epoch {epoch+1}/{total_epochs}")
    
    # Combine legends from both axes
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    fig.tight_layout()

    # Update the specific display slot in Jupyter
    if plot_handle:
        plot_handle.update(fig)
    else:
        display(fig)
        
    plt.close(fig) # Prevent memory leaks