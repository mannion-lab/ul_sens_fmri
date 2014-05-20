import os

import numpy as np
import PIL
import psychopy.visual
import psychopy.event
import psychopy.core

import stimuli.psychopy_ext

import ul_sens_fmri.config
import ul_sens_fmri.stim



def run(conf, subj_id, run_num, serial_port=None):

    (seq_log_path, task_log_path) = get_log_paths(conf, subj_id, run_num)

    run_seq = gen_run_seq(conf)

    frags = ul_sens_fmri.stim.get_img_fragments(conf)


    with stimuli.psychopy_ext.WindowManager(
        size=conf.exp.screen_size,
        monitor=conf.exp.monitor_name,
        lut=True
    ) as win:

        stim = ul_sens_fmri.stim.Stim(win=win, conf=conf, fragments=frags)

        # init the first trial
        for i_pres in xrange(2):
            img_id = run_seq[i_pres, 0, 2]

            if img_id == 0 or run_seq[i_pres, 0, 0] > 0:
                stim.set_contrast(conf.exp.pres_locs[i_pres], 0.0)
            else:
                src_vert_loc = conf.exp.src_locs[
                    int(run_seq[i_pres, 0, 1] - 1)
                ]
                stim.set_img(
                    img_id=img_id,
                    pres_vert_loc=conf.exp.pres_locs[i_pres],
                    src_vert_loc=src_vert_loc
                )
                stim.set_contrast(conf.exp.pres_locs[i_pres], 1.0)

            run_seq[i_pres, 0, 4] = 1

        run_clock = psychopy.core.Clock()

        run_clock.reset()

        keep_going = True

        while run_clock.getTime() < conf.exp.run_len_s and keep_going:

            stim.draw()

            win.flip()

            flip_time = run_clock.getTime() + (1.0 / 60.0)

            keys = psychopy.event.getKeys()

            for key in keys:
                keep_going = False

            for i_pres in xrange(2):
                # indice of the trial
                i_trial = np.where(flip_time >= run_seq[i_pres, :, 0])[0]

                if len(i_trial) == 0:
                    if flip_time < 2.0:
                        stim.set_contrast(conf.exp.pres_locs[i_pres], 0.0)
                        break
                    else:
                        raise ValueError("Timing error")
                else:
                    i_trial = i_trial[-1]

                trial_time = flip_time - run_seq[i_pres, i_trial, 0]

                if trial_time > conf.exp.stim_on_s:

                    if run_seq[i_pres, i_trial, 4] != 3:

                        stim.set_contrast(conf.exp.pres_locs[i_pres], 0.0)
                        run_seq[i_pres, i_trial, 4] = 3

                    if i_trial < conf.exp.n_run_trials:

                        if run_seq[i_pres, i_trial + 1, 4] == 0:

                            next_img_id = run_seq[i_pres, i_trial + 1, 2]
                            next_src_loc = conf.exp.src_locs[
                                int(run_seq[i_pres, i_trial + 1, 1] - 1)
                            ]

                            if next_img_id == 0:
                                stim.set_contrast(
                                    conf.exp.pres_locs[i_pres],
                                    0.0
                                )
                            else:
                                stim.set_img(
                                    img_id=next_img_id,
                                    pres_vert_loc=conf.exp.pres_locs[i_pres],
                                    src_vert_loc=next_src_loc
                                )

                            run_seq[i_pres, i_trial + 1, 4] = 1

                else:

                    if run_seq[i_pres, i_trial, 4] != 2:

                        if run_seq[i_pres, i_trial, 2] != 0:
                            contrast = 1.0
                        else:
                            contrast = 0.0

                        stim.set_contrast(conf.exp.pres_locs[i_pres], contrast)

                        run_seq[i_pres, i_trial, 4] = 2



def init_task(conf):
    """Initialises the task timing.

    Returns
    -------
    task_lut : numpy array, shape of ( evt x info )
        Task lookup table, where dim two is ( time_s, digit, polarity, target )
    targets : numpy array, shape of ( target, info )
        Target information, stored as ( digit, polarity )

    """

    n_task_per_run = int(
        conf.exp.run_len_s *
        conf.exp.task_rate_hz
    )

    task_set = conf.exp.task_set
    np.random.shuffle(task_set)

    targets = np.array(
        [
            [
                task_set[i],
                conf.exp.task_polarity[i]
            ]
            for i in xrange(2)
        ]
    )

    # second dim is (time, digit, polarity, target or not)
    task_lut = np.empty((n_task_per_run, 4))

    for i_evt in xrange(n_task_per_run):

        time_s = i_evt * (1.0 / conf.exp.task_rate_hz)

        curr_task_set = task_set.copy()
        curr_task_set = curr_task_set[curr_task_set != task_lut[i_evt - 1, 1]]

        digit = curr_task_set[np.random.randint(len(curr_task_set))]

        polarity = conf.exp.task_polarity[np.random.randint(2)]

        if np.any(
            np.logical_and(
                targets[:, 0] == digit,
                targets[:, 1] == polarity
            )
        ):
            target = 1
        else:
            target = 0

        task_lut[i_evt, :] = [time_s, digit, polarity, target]

    return (task_lut, targets)


def gen_run_seq(conf):
    """Generates a run sequence for the two presentation conditions.

    The returned array has three dimensions:
        presentation location (a, b)
        trial
        trial info:
            time, in seconds, for the start of the trial
            source location: 1 for 'a', 2 for 'b', 0 for null
            image id: the frame number of the image to show, or 0 for null
            is pre: 1 if is the pre events, 0 if the main experiment
            been prepped: 1 if the trial has been prepared, 0 if not

    """

    # this is presentation location (a, b) x trial x (time, src location,
    # image_id, is_pre, been_prepped)
    run_seq = np.empty((2, conf.exp.n_run_trials, 5))
    run_seq.fill(np.NAN)

    # one of the pres conditions gets a half-bin time offset
    time_offset = np.array([True, False])
    time_offset = time_offset[np.random.permutation(2)]

    for i_pres_loc in xrange(2):

        # generate the sequence of image IDs
        # two sequences because of difference source locations
        img_id_seq = np.tile(conf.exp.img_ids, conf.exp.n_src_locs)

        # join with zeros for the null trials
        img_id_seq = np.concatenate(
            (
                img_id_seq,
                np.zeros(conf.exp.n_null_trials)
            )
        )

        # source location indicator
        src_loc_seq = np.concatenate(
            (
                np.ones(conf.exp.n_img) * 1,
                np.ones(conf.exp.n_img) * 2,
                np.zeros(conf.exp.n_null_trials)
            )
        )

        # generate a random ordering
        i_order = np.random.permutation(conf.exp.n_seq_trials)

        # and apply
        img_id_seq = img_id_seq[i_order]
        src_loc_seq = src_loc_seq[i_order]

        # do some checks
        for img_id in conf.exp.img_ids:

            assert np.sum(img_id_seq == img_id) == conf.exp.n_src_locs

            assert all(
                [
                    np.sum(
                        np.logical_and(
                            img_id_seq == img_id,
                            src_loc_seq == n
                        )
                    ) == 1
                    for n in xrange(1, 3)
                ]
            )

        # to to add the wrapped trials at the beginning
        full_img_id_seq = np.concatenate(
            (
                img_id_seq[-conf.exp.n_pre_trials:],
                img_id_seq
            )
        )
        full_src_loc_seq = np.concatenate(
            (
                src_loc_seq[-conf.exp.n_pre_trials:],
                src_loc_seq
            )
        )

        is_pre = np.zeros(conf.exp.n_run_trials)
        is_pre[:conf.exp.n_pre_trials] = 1

        # now we can fill in our array
        run_seq[i_pres_loc, :, 1] = full_src_loc_seq
        run_seq[i_pres_loc, :, 2] = full_img_id_seq
        run_seq[i_pres_loc, :, 3] = is_pre

        run_seq[i_pres_loc, :, 0] = (
            np.arange(conf.exp.n_run_trials) *
            conf.exp.trial_len_s
        )

        if time_offset[i_pres_loc]:
            run_seq[i_pres_loc, :, 0] += conf.exp.trial_len_s / 2

        run_seq[i_pres_loc, :, 4] = 0

    assert np.sum(np.isnan(run_seq)) == 0

    return run_seq


def get_log_paths(conf, subj_id, run_num, err_if_exists=True):

    if not os.access(conf.exp.log_data_path, os.W_OK):
        raise ValueError("Cannot write to " + conf.exp.log_data_path)

    log_paths = []

    for log_type in ["seq", "task"]:

        log_file = "{s:s}_{e:s}_run_{n:02d}_{t:s}.txt".format(
            s=subj_id,
            e=conf.exp.exp_id,
            n=run_num,
            t=log_type
        )

        log_path = os.path.join(
            conf.exp.log_data_path,
            log_file
        )

        if os.path.exists(log_path):
            raise ValueError("Path " + log_path + " already exists")

        log_paths.append(log_path)

    return log_paths



