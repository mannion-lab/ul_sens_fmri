import os

import numpy as np


class ConfigContainer(object):
    pass


def get_conf(subj_id=None):

    conf = ConfigContainer()

    conf.stim = _get_stim_conf()
    conf.acq = _get_acq_conf()
    conf.exp = _get_exp_conf()
    conf.ana = _get_ana_conf()
    conf.all_subj = _get_subj_conf(subj_id=None)

    if subj_id is not None:
        conf.subj = _get_subj_conf(subj_id=subj_id)

    return conf


def _get_stim_conf():

    stim_conf = ConfigContainer()

    # get the location of the current code
    config_dir = os.path.dirname(os.path.realpath(__file__))
    # split into directories
    config_dir_split = config_dir.split(os.sep)
    # replace the last with the database link
    config_dir_split[-1] = "img_db"
    # and put back together
    stim_conf.img_db_path = os.sep.join(config_dir_split)

    stim_conf.decode_type = "edgesense"

    # total size of the frame to extract
    stim_conf.frame_size_pix = 432

    # size of each aperture
    # nice and power-of-two ish
    stim_conf.img_aperture_size_pix = 128

    stim_conf.img_aperture_locs_pix = {}

    # workout the image grid corresponding to each of the four image locations
    # this assumes that (0,0) is the upper left corner
    for (row_offset, row_loc) in zip([0.25, 0.75], ["a", "b"]):
        for (col_offset, col_loc) in zip([0.25, 0.75], ["l", "r"]):

            row_c = int(row_offset * stim_conf.frame_size_pix)
            col_c = int(col_offset * stim_conf.frame_size_pix)

            rows = np.arange(
                row_c - stim_conf.img_aperture_size_pix / 2,
                row_c + stim_conf.img_aperture_size_pix / 2
            )

            assert len(rows) == stim_conf.img_aperture_size_pix

            cols = np.arange(
                col_c - stim_conf.img_aperture_size_pix / 2,
                col_c + stim_conf.img_aperture_size_pix / 2
            )

            assert len(cols) == stim_conf.img_aperture_size_pix

            stim_conf.img_aperture_locs_pix[row_loc + col_loc] = [
                rows,
                cols
            ]


    return stim_conf


def _get_acq_conf():

    pass


def _get_exp_conf():

    exp_conf = ConfigContainer()

    exp_conf.environment = "garden"

    exp_conf.n_img = 30

    # replace this
    exp_conf.img_ids = np.arange(1, exp_conf.n_img + 1)

    exp_conf.vf_pos = ["al", "ar", "bl", "br"]

    return exp_conf


def _get_ana_conf():

    pass


def _get_subj_conf(subj_id=None):

    pass
