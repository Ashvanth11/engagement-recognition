import argparse
from pathlib import Path

import numpy as np
from keras.models import load_model
# GPU config if using gpu then uncomment following lines
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from input_pipeline import get_dataset


class_names = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

output_dir = "outputdir"
numpy_dir = "numpyoutput"
model_dir = "modelpath"

def print_accuracy(result):
    """Print accuracy for all the classes."""
    print("Accuracy: ")
    for i, acc in enumerate(result[5:]):
        print(f"{class_names[i]}: {acc*100}")


def test(dataset, model,  out_dir):
    """
    Evaluate test data on all the models.

    Args:
        dataset: tf.data.Dataset object as from input_pipeline
        model: Model instance with no finetune and no Dense layers.
        model_fine: Model instance with finetune and no Denselayers.
        model_fc: Model instance with no finetune and Dense layers.
        model_fc_fine: Model instance with finetune and Dense layers.
        out_dir: Directory to store results of evaluation.
    """
    if out_dir.parts[-1] == "fps_0.7":
        model_result = model.evaluate(dataset)
        np.save(f"{str(out_dir)}", np.array(model_result))

        print("=" * 50)
        print("=" * 50)
        print("Testing on data with frame rate as 0.7fps")
        
        print_accuracy(model_result)
        print("=" * 50)
        print("=" * 50)
    else:
        model_fc_result = model.evaluate(dataset)
        np.save(f"{str(out_dir)}", np.array(model_fc_result))
       
        print("=" * 50)
        print("=" * 50)
        print("Testing on data with default frame rate")
       
        print_accuracy(model_fc_result)
        print("=" * 50)
        print("=" * 50)


def evaluate(model_dir, out_dir):
    """
    Create neccessary directories, read the trained model and call
    test to evaluate.

    Args:
        model_dir: Directory which contains trained model.
        numpy_dir: Dirctory which contains filepath, label array.
        out_dir: Directory to store results of evaluation as array.
    """
    out_dir = Path(out_dir)
    odir1 = out_dir / "fps_0.7"
    odir1.mkdir(parents=True, exist_ok=True)
    odir2 = out_dir / "fps_default"
    odir2.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model = load_model(f"{str(model_dir)}")
    
    test_ds = get_dataset("Test")
    test_ds_final = get_dataset("FinalTest")

    #test(test_ds, model, odir1)
    test(test_ds_final, model, odir2)

evaluate(model_dir, output_dir)
