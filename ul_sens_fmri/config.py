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

    stim_conf.ap_size_deg = 4.5
    stim_conf.ap_ecc_deg = 5.37

    ap_offset_deg = stim_conf.ap_ecc_deg / np.sqrt(2.0)

    stim_conf.ap_pos_deg = {
        "al": [-ap_offset_deg, +ap_offset_deg],
        "ar": [+ap_offset_deg, +ap_offset_deg],
        "bl": [-ap_offset_deg, -ap_offset_deg],
        "br": [+ap_offset_deg, -ap_offset_deg]
    }

    return stim_conf


def _get_acq_conf():

    pass


def _get_exp_conf():

    exp_conf = ConfigContainer()

    exp_conf.log_data_path = "/sci/study/ul_sens_fmri/log_data"

    exp_conf.exp_id = "ul_sens_fmri"

    exp_conf.monitor_name = "BOLDscreen"
    exp_conf.screen_size = (1920, 1200)

    exp_conf.environment = "urban"

    exp_conf.n_img = 30

    exp_conf.pres_locs = ["a", "b"]

    # fragment source locations - above and below
    exp_conf.src_locs = ["a", "b"]

    exp_conf.n_src_locs = len(exp_conf.src_locs)

    # total number of event trials
    exp_conf.n_stim_trials = exp_conf.n_img * exp_conf.n_src_locs

    # proportion of 'null' trials
    exp_conf.null_prop = 0.25

    # number of 'null' trials
    exp_conf.n_null_trials = exp_conf.n_stim_trials * exp_conf.null_prop

    # check that number of null trials is an integer
    np.testing.assert_almost_equal(np.mod(exp_conf.n_null_trials, 1.0), 0.0)

    exp_conf.n_null_trials = int(exp_conf.n_null_trials)

    # total length of the run sequence
    exp_conf.n_seq_trials = exp_conf.n_stim_trials + exp_conf.n_null_trials

    # number of trials to wrap-around at the beginning
    exp_conf.n_pre_trials = 8

    # total number of trials in a run - sequence plus pre
    exp_conf.n_run_trials = exp_conf.n_seq_trials + exp_conf.n_pre_trials

    exp_conf.trial_len_s = 4.0

    exp_conf.run_len_s = exp_conf.n_run_trials * exp_conf.trial_len_s

    exp_conf.stim_on_s = 1.0

    # replace this
    exp_conf.img_ids = np.arange(1, exp_conf.n_img + 1)

    # this was generated via:
    #   random.choice(range(1,1001),30)
    exp_conf.img_ids = np.array(
        [
            856, 933, 352, 167, 806, 327, 489, 161, 383, 052,
            244, 338, 395, 797, 659, 224, 722, 314, 013, 376,
            840, 307, 971, 999, 241, 706, 001, 387, 600, 686
        ]
    )

    exp_conf.vf_pos = ["al", "ar", "bl", "br"]

    exp_conf.n_runs = 10

    exp_conf.stim_contrast_profile = np.empty((1000, 2))
    exp_conf.stim_contrast_profile.fill(np.NAN)

    # time lookup
    exp_conf.stim_contrast_profile[:, 0] = np.linspace(
        0,
        exp_conf.stim_on_s,
        1000
    )

    ramp_len_s = 0.1
    ramp_len = np.sum(exp_conf.stim_contrast_profile[:, 0] <= ramp_len_s)

    hann_win = np.hanning(ramp_len * 2)

    ramp = np.concatenate(
        (
            hann_win[:ramp_len],
            np.ones(1000 - ramp_len * 2),
            hann_win[ramp_len:]
        )
    )

    exp_conf.stim_contrast_profile[:, 1] = ramp

    exp_conf.task_set = np.arange(10)
    exp_conf.task_polarity = [-1, +1]
    exp_conf.task_rate_hz = 3.0

    return exp_conf


def _get_ana_conf():

    pass


def _get_subj_conf(subj_id=None):

    pass
