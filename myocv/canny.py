import cv2 as cv
from myocv.tb import Trackbar


class Canny(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Canny") -> None:
        super().__init__(src_img_path, name, init_val=128, max_val=255)
        self.blur = self.img_processor.blur(self.gray)

    def process_images(self):
        self.canny = cv.Canny(
            self.blur, self.trackbar_value, self.trackbar_value * 2)
        return self.img_processor.process(self.canny)
