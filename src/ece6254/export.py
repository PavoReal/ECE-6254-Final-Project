import os

def find_models_in_dir(model_dir):
    model_extension = 'keras';
    data_extension  = 'pkl';

    models = [];

    try:
        files_in_dir = os.listdir(model_dir)

    except:
        print(f"No files found in {model_dir}");

    return models;

def export_line_chart(models, export_dir):
    pass;

def export_bar_chart(models, export_dir):
    pass;

def export_main(model_dir, export_dir):

    models = find_models_in_dir(model_dir);

    if len(models) == 0:
        print(f"No models found in {model_dir}");
        return;

    export_line_chart(models, export_dir);
    export_bar_chart(models, export_dir);

