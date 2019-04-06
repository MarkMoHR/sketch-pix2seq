# sketch-pix2seq

This is the reimplementation code of paper [Sketch-pix2seq: a Model to Generate Sketches of Multiple Categories](https://arxiv.org/pdf/1709.04121.pdf).

| Input | Generated examples |
| --- | --- |
| ![output examples](https://github.com/MarkMoHR/sketch-pix2seq/blob/master/assets/1583-sample_gt.png) | ![output examples](https://github.com/MarkMoHR/sketch-pix2seq/blob/master/assets/1583-sample_pred_cond_100.svg) |


## Requirements

- Python 3
- Tensorflow (>= 1.4.0)
- [InkScape](https://inkscape.org/) or [CairoSVG](https://cairosvg.org/) (For vector sketch rendering. Choose one of them is ok.)

  ```
  sudo apt-get install inkscape
  # or
  pip3 install cairosvg
  ```


## Data Preparations

Follow these steps:

1. First download the `.npz` data from [*QuickDraw*](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset) dataset. Choose one or more categories as you like.

1. Place the `.npz` packages under `datasets/npz/` dir.

1. Modify the hyper params in `get_default_hparams()` in [model.py](https://github.com/MarkMoHR/sketch-pix2seq/blob/master/model.py)
    
    - Set the name(s) of the downloaded packages in `data_set`
    - Set the size of the raster image in `img_H` / `img_W`
    
1. We provide two approaches of rendering sequential data into raster images:

    - Using [InkScape](https://inkscape.org/): this approach is **slower** but **accurate** all the time
    
    ```
    python render_svg2bitmap.py --data_base_dir='datasets' --render_mode='v1'
    ```

    - Using [CairoSVG](https://cairosvg.org/): this approach is **faster**, but will have one-pixel **misalignment** problem sometimes (when setting image-width to 256, it will turn out to be 255 sometimes)

    ```
    python render_svg2bitmap.py --data_base_dir='datasets' --render_mode='v2'
    ```
    
    The raster images can be found under `datasets/` dir.


## Training

Run this command for default training:

```
python sketch_pix2seq_train.py
```

You can also change the settings, *e.g.* image size, batch size, in `get_default_hparams()` in [model.py](https://github.com/MarkMoHR/sketch-pix2seq/blob/master/model.py). For multi-category training, set the `data_set` with a list of more than one packages' names.

### Training procedure

The following figure shows the *reconstruction loss* during training within 60k iterations. The orange line belongs to [sketch-rnn](https://arxiv.org/abs/1704.03477) and the blue one belongs to sketch-pix2seq.

![loss](https://github.com/MarkMoHR/sketch-pix2seq/blob/master/assets/loss-r.png)


## Sampling

With trained model placed under `outputs/snapshot/` dir, run this command for sampling:

```
python sketch_pix2seq_sampling.py
```

And results in `.svg` format can be found under `outputs/sampling/` dir.


## Credits
- This code is modified from repo of [Sketch-RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn).


