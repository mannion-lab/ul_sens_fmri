import os

import numpy as np
import PIL

import ul_sens_fmri.config


def _get_img_fragments(conf):

    frags = np.empty(
        (
            conf.exp.n_img,
            2,  # upper/lower
            2,  # left/right
            conf.stim.img_aperture_size_pix,
            conf.stim.img_aperture_size_pix,
            3  # RGB
        )
    )
    frags.fill(np.NAN)

    frags = {}

    for img_id in conf.exp.img_ids:

        img_frags = {}

        img_path = os.path.join(
            conf.stim.img_db_path,
            conf.exp.environment,
            conf.stim.decode_type,
            "frame_{n:05d}.png".format(n=img_id)
        )

        img = np.array(PIL.Image.open(img_path))

        col_chop = (img.shape[1] - conf.stim.frame_size_pix) / 2

        img = img[
            :,
            col_chop:(img.shape[1] - col_chop),
            :
        ]

        assert img.shape == (
            conf.stim.frame_size_pix,
            conf.stim.frame_size_pix,
            3
        )

        for vert in ["a", "b"]:
            for horiz in ["l", "r"]:

                (rows, cols) = conf.stim.img_aperture_locs_pix[vert + horiz]

                # if not done this way, bad stuff happens with broadcasting
                img_frag = img[
                    rows[0]:(rows[-1] + 1),
                    cols[0]:(cols[-1] + 1),
                    :
                ]

                img_frag = img_frag / 255.0 * 2.0 - 1.0

                img_frags[vert + horiz] = img_frag

        frags[img_id] = img_frags

    return frags

