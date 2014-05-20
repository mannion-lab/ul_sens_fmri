import os

import numpy as np
import PIL

import psychopy.visual


class Stim(object):

    def __init__(
        self,
        win,
        conf,
        fragments
    ):

        self._win = win
        self._conf = conf
        self._fragments = fragments

        self._textures = {}

        for vert in ["a", "b"]:
            for horiz in ["l", "r"]:

                tex = psychopy.visual.GratingStim(
                    win=self._win,
                    size=[conf.stim.ap_size_deg] * 2,
                    units="deg",
                    interpolate=True,
                    sf=1.0 / conf.stim.ap_size_deg,
                    mask="raisedCos",
                    pos=conf.stim.ap_pos_deg[vert + horiz]
                )

                self._textures[vert + horiz] = tex

    def set_img(self, img_id, pres_vert_loc, src_vert_loc):

        for horiz in ["l", "r"]:

            if img_id == 0:
                img = np.zeros((4,4))
            else:
                img = np.flipud(
                    self._fragments[img_id][src_vert_loc + horiz]
                )

            self._textures[pres_vert_loc + horiz].setTex(img)

    def set_contrast(self, vert_loc, contrast):

        for horiz in ["l", "r"]:
            self._textures[vert_loc + horiz].setContrast(
                contrast
            )

    def draw(self):

        for vert in ["a", "b"]:
            for horiz in ["l", "r"]:
                self._textures[vert + horiz].draw()


def get_img_fragments(conf):

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

                # these are store un-vertically flipped - they will need to be
                # flipped before use in PsychoPy
                img_frags[vert + horiz] = img_frag

        frags[img_id] = img_frags

    return frags
