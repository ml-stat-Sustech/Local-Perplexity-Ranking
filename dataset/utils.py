from .generation_dataset import GenerationTasksDatasets

def get_dataloader(task, type, noise_type, noise_ratio, seed):

    raw_dataset = GenerationTasksDatasets(dataset_type=type, task=task, noise_type=noise_type, noise_ratio=noise_ratio, seed=seed)
        
    return raw_dataset
    