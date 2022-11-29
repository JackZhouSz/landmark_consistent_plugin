from distutils import extension
from functools import partial
from collections import namedtuple
import os
import json
import numpy as np
import random


from fire import Fire
from pytorch_toolbelt.utils import read_rgb_image

from predictor import FaceMeshPredictor
from demo_utils import (
    draw_landmarks,
    draw_3d_landmarks,
    draw_mesh,
    draw_pose,
    get_mesh,
    get_flame_params,
    get_output_path,
    MeshSaver,
    ImageSaver,
    JsonSaver,
)

from model_training.metrics.keypoints import KeypointsNME512 as KeypointsNME
from torchmetrics import MetricCollection
metrics_2d = MetricCollection(
    {
        "nme_2d": KeypointsNME(compute_on_step=True),
    }
)
metrics_2d_dad3d = MetricCollection(
    {
        "nme_2d": KeypointsNME(compute_on_step=True),
    }
)

DemoFuncs = namedtuple(
    "DemoFuncs",
    ["processor", "saver"],
)

demo_funcs = {
    "68_landmarks": DemoFuncs(draw_landmarks, ImageSaver),
    "247_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="247"), ImageSaver),
    "191_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="191"), ImageSaver),
    "445_landmarks": DemoFuncs(partial(draw_3d_landmarks, subset="445"), ImageSaver),
    "head_mesh": DemoFuncs(partial(draw_mesh, subset="head"), ImageSaver),
    "face_mesh": DemoFuncs(partial(draw_mesh, subset="face"), ImageSaver),
    "pose": DemoFuncs(draw_pose, ImageSaver),
    "3d_mesh": DemoFuncs(get_mesh, MeshSaver),
    "flame_params": DemoFuncs(get_flame_params, JsonSaver)
}

def demo(
   input_image_path: str = 'images',
   outputs_folder: str = "outputs",
   type_of_output: str = "68_landmarks",
) -> None:

   experiments_key = 1
   images_names = sorted(os.listdir(input_image_path))
   experiments_dict = {0:{'ext': 'dad3d', \
                        'model_path': 'checkpoints/dad3d/epoch_99-step_315299-vlast.trcd'}, \
                     1:{'ext': 'dad3d+', \
                        'model_path': 'checkpoints/dad3d+/epoch_99-step_26199-vlast.trcd'}, \
                     }
   outputs_path = os.path.join(outputs_folder, input_image_path)
   os.makedirs(outputs_path, exist_ok=True)

   # Preprocess and get predictions.
   predictor = FaceMeshPredictor.dad_3dnet(model_path=os.path.join(os.getcwd(), experiments_dict[experiments_key]['model_path']))
   cnt = 0
   for i in range(len(images_names)):
      image_name = images_names[i]
      image_path = os.path.join(input_image_path, image_name)
      image = read_rgb_image(image_path)
      predictions = predictor(image)
      
      # Get the resulting output.
      result = demo_funcs[type_of_output].processor(predictions, image)
      # Save the demo output.
      saver = demo_funcs[type_of_output].saver()  # instantiate the Saver
      output_img_path = os.path.join(outputs_path, image_name.replace('.png', '_{}.jpg'.format(experiments_dict[experiments_key]['ext'])))
      saver(result, output_img_path)
            
      print('cnt: {}/{}'.format(str(cnt).zfill(4), str(len(images_names)).zfill(4)))
      # if cnt >= image_num:
      #     break
      cnt += 1
      # pdb.set_trace()


if __name__ == "__main__":
   Fire(demo)
