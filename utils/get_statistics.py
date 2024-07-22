import matplotlib.pyplot as plt 



def get_activation_statistics(activation_dict, save_fig):
    """
    Plots statistics of the activations of a given layer, the statistics include a histogram and a heatmap.
    The histogram shows the distribution of the activation values of the given layer.
    The heatmap shows the number of saturated neurons of the given layer.
      
    Arguments:
        - activation_dict: Dict which should contain the activations of the model. Each key corresponds to a layer and each value contains a list of tuples.
        The tuples contain the raw activations of the model as well as the number of steps trained.
        - save_fig: Select if the figures of the statistics should be saved.
    
        Returns:
        - None

    """

    for layer_name, layer_activations in activation_dict.items():
        for activation_steps in layer_activations:

            
            activation = activation_steps[0]
            step = activation_steps[1]

            if isinstance(step, str):
                pass
            else:
                step = str(step)
            step_name = "Step_" + step


            activation_1d = activation.view(-1).tolist()
            
            # histogram of activations after layer
            plt.figure()
            plt.hist(activation_1d, bins=50)
            plt.title(f"Histogram of {layer_name} Activations at {step_name}")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            if save_fig:
                plt.savefig(f"Actvation_Histogram_{step_name}.png")
            else:
                plt.show()

            # heatmap of activations after layer
            plt.figure()
            plt.imshow(activation.abs() > 0.99, cmap="gray", interpolation="nearest") # show saturated activations of all neurons in white
            plt.title(f"Heat Map of {layer_name} Activations at {step_name}")        # if a whole column is white, the neuron is always saturated which destroys the gradient flow -> bad!
            plt.xlabel(f"Neurons in {layer_name}")
            plt.ylabel("Batches")
            if save_fig:
                plt.savefig(f"Actvation_Heatmap_{step_name}.png")
            else:
                plt.show()
            