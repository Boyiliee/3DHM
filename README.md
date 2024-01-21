# Synthesizing Moving People with 3D Control

Please [take a look at our demo video](https://www.youtube.com/watch?v=iGGlFQJ-OyQ) for a brief overview.

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

## Updates

Thanks for your interest in 3DHM! We will release our code and related resources shortly.

