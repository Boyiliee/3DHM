# Synthesizing Moving People with 3D Control

Please [take a look at our demo video](https://www.youtube.com/watch?v=obABCBP-fC0) for a brief overview.

## Abstract
We present a diffusion model-based framework for animating people from a single image for a given target 3D motion sequence.

Our approach has two core components: a) learning priors about invisible parts of the human body and clothing, and b) rendering novel body poses with proper clothing and texture. For the first part, we learn an in-filling diffusion model to hallucinate unseen parts of a person given a single image. We train this model on texture map space, which makes it more sample-efficient since it is invariant to pose and viewpoint. Second, we develop a diffusion-based rendering pipeline, which is controlled by 3D human poses. This produces realistic renderings of novel poses of the person, including clothing, hair, and plausible in-filling of unseen regions.

This disentangled approach allows our method to generate a sequence of images that are faithful to the target motion in the 3D pose and, to the input image in terms of visual similarity. In addition to that, the 3D control allows various synthetic camera trajectories to render a person. Our experiments show that our method is resilient in generating prolonged motions and varied challenging and complex poses compared to prior methods.

## 3DHM Training Features
💡 3DHM training pipeline (for both stages) is self-supervised. <br>
💡 3DHM does not use any additional annotations. It is trained with pseudo-ground-truth as we use cutting-edge software which can detect, segment, track, and 3Dfy humans (H4D). <br>
💡 3DHM is scalable and its scaling can be done readily in the future given additional videos of humans in motion and computing resources.

## 3DHM Key Features 
💡 Various Camera Viewpoints. <br>
💡 Motions from Text. <br>
💡 Motions from Random Videos (Various 3D poses). <br>
💡 Various Camera Azimuths. <br>
💡 Long-range Motions. <br>
💡 Challenging Motions. <br>
💡 Animations from just the Back View.

Please [visit our website](https://boyiliee.github.io/3DHM.github.io/) for more information!

## Installation
You should follow the link to install [4D-Humans](https://github.com/shubham-goel/4D-Humans) as an conda environment. Then build 3dhm environment by:

`
conda env create -f 3dhm.yml
`

You also need to download our model weights on [Huggingface](https://huggingface.co/Leooo333/3DHM) and put them under `./models`


## Gradio DEMO

`
python app.py
`

## Inference

### 1. Data process (4D-Humans)

Install 4D-Humans and then activate the environments

`
conda activate 4D-Humans
`

List the file structure like this

```text
|-- driving_videos
    |-- your_video_1
    	|-- video.mp4
    |-- your_video_2
        |-- video.mp4
    ...

|-- reference_imgs
    |-- images
        |-- your_ref_img_A.png
        |-- your_ref_img_B.png
                ...
```

Run data preprocess:

`
python track.py driving_videos=ABS_driving_videos_folder reference_imgs=ABS_reference_imgs_folder
`

You will get:

```text
|-- driving_videos
    |-- your_video_1
        |-- results
        	|-- eval_.pkl
        |-- mask.pkl
        |-- texture.png
    ...

|-- reference_imgs
    |-- your_ref_img_A
        |-- results
        	|-- eval_.pkl
        |-- mask.pkl
        |-- texture.png

|-- processed_datas.json

```

The generated processed_datas.json cantains processed reference and motion pairs, you can select desired pair by manually modified the json file.


### 2. Stage-1 inpainting Model

`
python stage1_inpaint.py reference_imgs=ABS_reference_imgs_folder
`

```text
|-- reference_imgs
    |-- your_ref_img_A
        |-- texture.png
```

It will replace the texture.png with inpainted one.


### 3. Stage-2 video generation

```
python smt/eval.py -m --config-name eval.yaml task_name="test_0" trainer=deepspeed \
configs.sdiff.task_name="test_0" \
configs.wights_path="models/stage2.pt" \
logger.tensorboard.save_dir="stage2_test/test_0" \
configs.sdiff.video_save_path="stage2_test/test_0" \
configs.test_batch_size=1 \
configs.test_num_workers=1 configs.base_model="stablemotion_video_v2" \
configs.timesteps_max=20 \
configs.sdiff.guidance_scale_control=2.5 \
configs.metadata_path="YOUR_processed_datas.json"
```
