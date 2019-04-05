import os
import subprocess
import argparse
import numpy as np
from PIL import Image

import model as sketch_rnn_model
from sketch_pix2seq_train import load_dataset
from sketch_pix2seq_sampling import draw_strokes


def svg2png(input_path, svgsize, pngsize, output_path, padding=False, padding_args="--export-area-drawing"):
    """
    convert .svg into .png
    :param input_path:
    :param svgsize: (w, h)
    :param pngsize: (w, h)
    :param output_path:
    :param padding: whether do padding to png
    :param padding_args:
    :return:
    """
    x_scale = pngsize[0] / svgsize[0]
    y_scale = pngsize[1] / svgsize[1]
    w, h = pngsize

    if x_scale > y_scale:
        y = int(h)
        cmd = "inkscape {0} {1} -e {2} -h {3}".format(input_path, padding_args, output_path, y)
    else:
        x = int(w)
        cmd = "inkscape {0} {1} -e {2} -w {3}".format(input_path, padding_args, output_path, x)

    # Do the actual rendering
    subprocess.call(cmd.split(), shell=False)

    if padding:
        curr_png = Image.open(output_path).convert('RGB')
        png_w = curr_png.width
        png_h = curr_png.height
        assert png_w == pngsize[0] or png_h == pngsize[1]

        max_dim = max(png_w, png_h)
        padded_png = np.zeros(shape=[max_dim, max_dim, 3], dtype=np.uint8)
        padded_png.fill(255)

        if png_w > png_h:
            pad = int(round((png_w - png_h) / 2))
            padded_png[pad: pad + png_h, :, :] = np.array(curr_png, dtype=np.uint8)
        else:
            pad = int(round((png_h - png_w) / 2))
            padded_png[:, pad: pad + png_w, :] = np.array(curr_png, dtype=np.uint8)

        padded_png = Image.fromarray(padded_png, 'RGB')
        padded_png.save(output_path, 'PNG')


def main(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    npz_dir = os.path.join(data_base_dir, 'npz')
    svg_dir = os.path.join(data_base_dir, 'svg')
    png_dir = os.path.join(data_base_dir, 'png')

    model_params = sketch_rnn_model.get_default_hparams()
    for dataset_i in range(len(model_params.data_set)):
        assert model_params.data_set[dataset_i][-4:] == '.npz'
        cate_svg_dir = os.path.join(svg_dir, model_params.data_set[dataset_i][:-4])
        cate_png_dir = os.path.join(png_dir, model_params.data_set[dataset_i][:-4])

        datasets = load_dataset(data_base_dir, model_params)

        data_types = ['train', 'valid', 'test']
        for d_i, data_type in enumerate(data_types):
            split_cate_svg_dir = os.path.join(cate_svg_dir, data_type)
            split_cate_png_dir = os.path.join(cate_png_dir, data_type,
                                              str(model_params.img_H) + 'x' + str(model_params.img_W))

            os.makedirs(split_cate_svg_dir, exist_ok=True)
            os.makedirs(split_cate_png_dir, exist_ok=True)

            split_dataset = datasets[d_i]

            for ex_idx in range(len(split_dataset.strokes)):
                stroke = np.copy(split_dataset.strokes[ex_idx])
                print('example_idx', ex_idx, 'stroke.shape', stroke.shape)

                png_path = split_dataset.png_paths[ex_idx]
                assert split_cate_png_dir == png_path[:len(split_cate_png_dir)]
                actual_idx = png_path[len(split_cate_png_dir) + 1:-4]
                svg_path = os.path.join(split_cate_svg_dir, str(actual_idx) + '.svg')

                svg_size = draw_strokes(stroke, svg_path, padding=10)  # (w, h)
                svg2png(svg_path, svg_size, (model_params.img_W, model_params.img_H), png_path,
                        padding=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', '-db', type=str, default='datasets', help="set the data base dir")
    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_base_dir,
    }

    main(**run_params)
