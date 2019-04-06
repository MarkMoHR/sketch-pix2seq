import numpy as np
import os
import json
import argparse
import tensorflow as tf
from six.moves import range
import svgwrite

import model as sketch_rnn_model
import utils
from sketch_pix2seq_train import load_dataset, reset_graph, load_checkpoint


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def draw_strokes(data, svg_filename, factor=0.2, padding=50):
    """
    little function that displays vector images and saves them to .svg
    :param data:
    :param factor:
    :param svg_filename:
    :param padding:
    :return:
    """
    min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
    dims = (padding + max_x - min_x, padding + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = int(padding / 2) - min_x
    abs_y = int(padding / 2) - min_y
    p = "M%s, %s " % (abs_x, abs_y)
    # use lowcase for relative position
    command = "m"
    for i in range(len(data)):
        if lift_pen == 1:
            command = "m"
        elif command != "l":
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + ", " + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()

    return dims, dwg.tostring()


def make_grid_svg(s_list, grid_space=20.0, grid_space_x=20.0):
    """
    generate a 2D grid of many vector drawings
    :param s_list:
    :param grid_space:
    :param grid_space_x:
    :return:
    """

    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def load_env_compatible(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    return load_dataset(data_dir, model_params, inference_mode=True)


def load_model_compatible(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = sketch_rnn_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def encode(input_images, session, model):
    unused_strokes = np.zeros(shape=[model.hps.batch_size, model.hps.max_seq_len + 1, 5], dtype=np.float32)
    return session.run(model.batch_z,
                       feed_dict={model.input_data: unused_strokes,
                                  model.input_image: input_images})[0]


def decode(session, sample_model, max_seq_len, z_input=None, temperature=0.1):
    z = None
    if z_input is not None:
        z = [z_input]

    sample_strokes, m = sketch_rnn_model.sample(session, sample_model,
                                                seq_len=max_seq_len, temperature=temperature, z=z)
    strokes = utils.to_normal_strokes(sample_strokes)  # sample_strokes in stroke-5 format, strokes in stroke-3 format
    return strokes


def sampling_conditional(data_dir, sampling_dir, model_dir):
    [train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = \
        load_env_compatible(data_dir, model_dir)

    # construct the sketch-rnn model here:
    reset_graph()
    model = sketch_rnn_model.Model(hps_model)
    eval_model = sketch_rnn_model.Model(eval_hps_model, reuse=True)
    sampling_model = sketch_rnn_model.Model(sample_hps_model, reuse=True)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, model_dir)

    for _ in range(20):
        # get a sample drawing from the test set, and render it to .svg
        stroke, rand_idx, image = test_set.random_sample()  # ndarray, [N_points, 3]
        sub_sampling_dir = os.path.join(sampling_dir, str(rand_idx))
        os.makedirs(sub_sampling_dir, exist_ok=True)
        print('rand_idx', rand_idx, 'stroke.shape', stroke.shape)
        draw_strokes(stroke, os.path.join(sub_sampling_dir, 'sample_gt.svg'))

        z = encode(image, sess, eval_model)
        strokes_out = decode(sess, sampling_model, eval_model.hps.max_seq_len, z, temperature=0.1)  # in stroke-3 format
        draw_strokes(strokes_out, os.path.join(sub_sampling_dir, 'sample_pred_cond.svg'))

        # Create generated grid at various temperatures from 0.1 to 1.0
        stroke_list = []
        for i in range(10):
            for j in range(3):
                stroke_list.append(
                    [decode(sess, sampling_model, eval_model.hps.max_seq_len, z, temperature=0.1), [j, i]])
        stroke_grid = make_grid_svg(stroke_list)
        draw_strokes(stroke_grid, os.path.join(sub_sampling_dir, 'sample_pred_cond_100.svg'))


def main(**kwargs):
    data_dir_ = kwargs['data_dir']
    model_dir_ = kwargs['model_dir']
    sampling_dir_ = kwargs['sampling_dir']
    os.makedirs(sampling_dir_, exist_ok=True)

    sampling_conditional(data_dir_, sampling_dir_, model_dir_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-db', type=str, default='datasets', help="set the data base dir")
    parser.add_argument('--model_dir', '-md', type=str, default='outputs/snapshot', help="set the trained model dir")
    parser.add_argument('--sampling_dir', '-sd', type=str, default='outputs/sampling', help="set the results dir")
    args = parser.parse_args()

    run_params = {
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "sampling_dir": args.sampling_dir,
    }

    main(**run_params)
