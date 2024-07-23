import os
import matplotlib.pyplot as plt 

from typing import Dict, List, Tuple


def plot_heatmaps(activation_dict:Dict[str,List[Tuple[float,int]]], 
                  save_fig:bool = False) -> None:
    """
    Creates heatmaps which show how many of the neurons of a given layer are saturated.
    The dimensions of the heatmaps are defined by the number of inputs and the number of neurons within this layer.
    The heatmaps are created for all given layers as well as all given steps.

    Arguments:
        - activation_dict: Should contain the activations of the model. Each key corresponds to a layer and each value contains a list of tuples.
        The tuples contain the raw activations of the model as well as the number of steps trained at this point.
        - save_fig: Select if the figures of the statistics should be saved.
    
    Returns:
        - None
    """
    for layer_name, layer_activations in activation_dict.items():
            for activation_steps in layer_activations:
        
                activation = activation_steps[0]
                step = str(activation_steps[1])
                step_name = "Step_" + step
                
                plt.figure()
                plt.imshow(activation.abs() > 0.99, cmap="gray", interpolation="nearest")
                plt.title(f"Heat Map of {layer_name} Activations at {step_name}")       
                plt.xlabel(f"Neurons in {layer_name}")
                plt.ylabel("Batches")

                if save_fig:
                    file_name = f"Actvation_Heatmap_{layer_name}_{step_name}.png"
                    save_or_show_plot(save_fig, file_name)
                else:
                    plt.show()


def plot_single_hist(activation_dict:Dict[str,List[Tuple[float,int]]],
                     save_fig:bool = False) -> None:
    """
    Creates single-layer histograms which show the distribution of the activations of the neurons of a given layer.
    The single-layer histograms are created for each layer as well as all given steps.

    Arguments:
        - activation_dict: Should contain the activations of the model. Each key corresponds to a layer and each value contains a list of tuples.
        The tuples contain the raw activations of the model as well as the number of steps trained at this point.
        - save_fig: Select if the figures of the statistics should be saved.
    
    Returns:
        - None
    """
    for layer_name, layer_activations in activation_dict.items():
            for activation_steps in layer_activations:

                activation = activation_steps[0]
                activation_1d = activation.view(-1).tolist()
                step = str(activation_steps[1])
                step_name = "Step_" + step
                
                plt.figure()
                plt.hist(activation_1d, bins=50)
                plt.title(f"Histogram of {layer_name} Activations at {step_name}")
                plt.xlabel("Activation Value")
                plt.ylabel("Frequency")

                if save_fig:
                    file_name = f"Actvation_Histogram_{layer_name}_{step_name}.png"
                    save_or_show_plot(save_fig, file_name)
                else:
                    plt.show()


def plot_multi_hist(activation_dict:Dict[str,List[Tuple[float,int]]],
                    save_fig:bool = False) -> None:
    """
    Creates multi-layer histograms which show the distribution of the activations of the neurons of all given layers in a single diagram.
    The multi-layer histograms are created for all steps.

    Arguments:
        - activation_dict: Should contain the activations of the model. Each key corresponds to a layer and each value contains a list of tuples.
        The tuples contain the raw activations of the model as well as the number of steps trained at this point.
        - save_fig: Select if the figures of the statistics should be saved.
    
    Returns:
        - None
    """
    steps_dict = {}
    layer_names = []
    for layer_name, layer_activations in activation_dict.items():
        layer_names.append(layer_name)
        for activation, step in layer_activations:

            if step not in steps_dict:
                steps_dict[step] = []
            steps_dict[step].append(activation.view(-1).tolist())

    for step, activations_list in sorted(steps_dict.items()):
        plt.figure()
        plt.hist(activations_list, bins=100, histtype="step", label=layer_names, linewidth=2)
        plt.title(f"Combined Histogram of All Layer Activations at Step {step}")
        plt.xlabel("Activation Value")
        plt.ylabel("Density")
        plt.legend(loc="upper center")
        
        if save_fig:
            file_name = f"Combined_Activation_Histogram_Step_{step}.png"
            save_or_show_plot(save_fig, file_name)
        else:
            plt.show()


def save_or_show_plot(file_name:str,
                      save_fig:bool = False) -> None:
    """
    Either displays or saves a matplotlib.pyplot figure.

    Arguments:
        - save_fig: Select if the figures of the statistics should be saved.
        - file_name: The figure will be saved under this name.

    Returns:
        - None
    """
    if save_fig:
        parent_dir, _ = os.path.split(os.getcwd())
        save_dir = os.path.join(parent_dir, "activation_statistics")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)

    else:
        plt.show()










            

