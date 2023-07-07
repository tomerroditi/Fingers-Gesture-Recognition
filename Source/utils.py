from pathlib import Path
import torch
import pickle


def save_model_and_pipeline(model, pipe, subject_num):
    root_path = Path.cwd().parent / 'models'
    # get the models directory path and save the model
    model_name = 'model_0'
    model_path = root_path / f'{subject_num:03d}/{model_name}.pth'
    if not model_path.parent.exists():
        # make sure that the directory exists
        model_path.parent.mkdir(parents=True)
    while model_path.exists():
        model_name = f'{model_name.split("_")[0]}_{int(model_name.split("_")[1])+1}'
        model_path = root_path / f'{subject_num:03d}/{model_name}.pth'
    torch.save(model, model_path)
    print(f'model saved to: {model_path}')
    # save the pipeline
    pipe_path = root_path / f'{subject_num:03d}/{model_name}_pipeline.pkl'
    with open(pipe_path, 'wb') as f:
        pickle.dump(pipe, f)
    print(f'pipeline saved to: {pipe_path}')

