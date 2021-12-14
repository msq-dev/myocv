import cv2 as cv
from myocv.tb import Trackbar


class OtsuThreshold(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Threshold Otsu") -> None:
        super().__init__(src_img_path, name, init_val=0, max_val=255)
        self.blur = self.img_processor.blur(self.gray)

    def process_images(self):
        _, self.otsu = cv.threshold(self.blur, self.init_val, self.max_val,
                                    cv.THRESH_BINARY+cv.THRESH_OTSU)
        return self.img_processor.process(self.otsu)
