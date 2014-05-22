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


class Fixation(object):

    def __init__(
        self,
        win,
        conf
    ):

        self._win = win
        self._conf = conf

        self._init_stim()


    def _init_stim(self):

        h_frac = 0.625

        l_s = [(+1, +0), (+h_frac, +1), (+0, +1), (-h_frac, +1)]
        l_e = [(-1, +0), (-h_frac, -1), (+0, -1), (+h_frac, -1)]

        self._grids = [
            psychopy.visual.Line(
                win=self._win,
                start=ls,
                end=le,
                units="norm",
                lineColor=[-0.25] * 3,
                lineWidth=1.5
            )
            for (ls, le) in zip(l_s, l_e)
        ]


        grids_r_va = [ 0.5, 1.8, 3.5, 6.1, 8.5 ]

        grid_lum = -0.25

        # list of circle stimuli
        self._rings = [
            psychopy.visual.Circle(
                win=self._win,
                radius=grid_r_va,
                units="deg",
                edges=96,
                lineColor=grid_lum,
                lineWidth=1.5
            )
            for grid_r_va in grids_r_va
        ]

        self._outlines = [
            psychopy.visual.Circle(
                win=self._win,
                pos=self._conf.stim.ap_pos_deg[ap],
                units="deg",
                edges=128,
                radius=self._conf.stim.ap_size_deg / 2.0,
                lineColor=[-0.25] * 3,
                fillColor=[0] * 3,
                lineWidth=1.5
            )
            for ap in ["al", "ar", "bl", "br"]
        ]

        self._fixation = psychopy.visual.Circle(
            win=self._win,
            radius=0.1,
            units="deg",
            fillColor=[-1, -1, -1],
            lineColor=[-0.25] * 3,
            lineWidth=2
        )

    def draw(self, draw_grid=True, draw_outlines=True, draw_centre=True):

        if draw_grid:
            _ = [grid.draw() for grid in self._grids]
            _ = [ring.draw() for ring in self._rings]

        if draw_outlines:
            _ = [outline.draw() for outline in self._outlines]

        if draw_centre:
            self._fixation.draw()

    def set_centre_polarity(self, polarity):

        self._fixation.setFillColor(polarity)


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
