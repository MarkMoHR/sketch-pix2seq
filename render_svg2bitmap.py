import os
import subprocess
import argparse
import numpy as np
from PIL import Image
import cairosvg

import model as sketch_rnn_model
from sketch_pix2seq_train import load_dataset
from sketch_pix2seq_sampling import draw_strokes


def pad_image(png_filename, pngsize, version):
    curr_png = Image.open(png_filename).convert('RGB')
    png_curr_w = curr_png.width
    png_curr_h = curr_png.height
    if version == 'v1':
        assert png_curr_w == pngsize[0] or png_curr_h == pngsize[1]
    else:
        if png_curr_w != pngsize[0] and png_curr_h != pngsize[1]:
            print('Not aligned', 'png_curr_w', png_curr_w, 'png_curr_h', png_curr_h)

    padded_png = np.zeros(shape=[pngsize[1], pngsize[0], 3], dtype=np.uint8)
    padded_png.fill(255)

    if png_curr_w > png_curr_h:
        pad = int(round((png_curr_w - png_curr_h) / 2))
        padded_png[pad: pad + png_curr_h, :png_curr_w, :] = np.array(curr_png, dtype=np.uint8)
    else:
        pad = int(round((png_curr_h - png_curr_w) / 2))
        padded_png[:png_curr_h, pad: pad + png_curr_w, :] = np.array(curr_png, dtype=np.uint8)

    padded_png = Image.fromarray(padded_png, 'RGB')
    padded_png.save(png_filename, 'PNG')


def svg2png_v1(input_path, svgsize, pngsize, png_filename, padding=False, padding_args="--export-area-drawing"):
    """convert svg into png, using inkscape"""
    svg_w, svg_h = svgsize
    png_w, png_h = pngsize
    x_scale = png_w / svg_w
    y_scale = png_h / svg_h

    if x_scale > y_scale:
        y = int(png_h)
        cmd = "inkscape {0} {1} -e {2} -h {3}".format(input_path, padding_args, png_filename, y)
    else:
        x = int(png_w)
        cmd = "inkscape {0} {1} -e {2} -w {3}".format(input_path, padding_args, png_filename, x)

    # Do the actual rendering
    subprocess.call(cmd.split(), shell=False)

    if padding:
        pad_image(png_filename, pngsize, 'v1')


def svg2png_v2(dwg_string, svgsize, pngsize, png_filename, padding=False):
    """convert svg into png, using cairosvg"""
    svg_w, svg_h = svgsize
    png_w, png_h = pngsize
    x_scale = png_w / svg_w
    y_scale = png_h / svg_h

    if x_scale > y_scale:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_height=png_h)
    else:
        cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_width=png_w)

    if padding:
        pad_image(png_filename, pngsize, 'v2')


def main(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    render_mode = kwargs['render_mode']

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

                svg_size, dwg_bytestring = draw_strokes(stroke, svg_path, padding=10)  # (w, h)

                if render_mode == 'v1':
                    svg2png_v1(svg_path, svg_size, (model_params.img_W, model_params.img_H), png_path, padding=True)
                elif render_mode == 'v2':
                    svg2png_v2(dwg_bytestring, svg_size, (model_params.img_W, model_params.img_H), png_path,
                               padding=True)
                else:
                    raise Exception('Error: unknown rendering mode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', '-db', type=str, default='datasets', help="set the data base dir")
    parser.add_argument('--render_mode', '-rm', type=str, choices=['v1', 'v2'], default='v1',
                        help="choose a rendering mode")
    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_base_dir,
        "render_mode": args.render_mode,
    }

    main(**run_params)
