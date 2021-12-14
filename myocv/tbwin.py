import cv2 as cv
import numpy as np
from myocv.tb import Trackbar

WIN_NAME = "SHOW"


class TrackbarWindow:
    def __init__(self, trackbars: list[Trackbar]) -> None:
        self.trackbars = trackbars

    def trackbar_callback(self, val) -> None:
        images = []
        for t in self.trackbars:
            t.get_position(WIN_NAME)
            images += t.process_images()

        window = self.make_window(images)
        cv.imshow(WIN_NAME, window)

    def show(self) -> None:
        cv.namedWindow(WIN_NAME)

        for t in self.trackbars:
            t.create(WIN_NAME, self.trackbar_callback)

        self.trackbar_callback(0)
        cv.waitKey(0)

    @staticmethod
    def make_window(images: list):
        for img in images:
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        if len(images) == 9:
            row_length = 3
        elif len(images) == 12:
            row_length = 4
        elif len(images) == 16:
            row_length = 4
        else:
            if len(images) % 2 != 0:
                img_width = images[0].shape[0]
                img_height = images[0].shape[1]
                filler = np.zeros((img_width, img_height, 3), np.uint8)
                filler[:] = (128, 128, 128)
                images.append(filler)
            row_length = len(images) // 2

        rows = [images[i:i + row_length]
                for i in range(0, len(images), row_length)]
        all = cv.vconcat(tuple([cv.hconcat(tuple(row)) for row in rows]))

        return all
