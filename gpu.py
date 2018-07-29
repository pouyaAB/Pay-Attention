class GPU():
    """
    This class is being used to controll the number of GPU
    used for the training process. GPU IDs can be added to
    'gpus_to_use'. The training process will take care of 
    the rest.
    The 'main_gpu' is where the central model has been stored.
    All the gradients from other models in other GPUs will be 
    sent to the 'main_gpu' and then copied back to all the GPUs
    after updating the weights.
    """
    gpus_to_use = [1, 0]
    num_gpus = len(gpus_to_use)
    main_gpu = gpus_to_use[0]
