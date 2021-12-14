import cv2 as cv
from myocv.tb import Trackbar


class SimpleThreshold(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Thresh",
                 threshold_type: str = "binary") -> None:
        self.threshold_types = {
            "binary": cv.THRESH_BINARY,
            "tozero": cv.THRESH_TOZERO
        }
        super().__init__(src_img_path, name=f"{name} {threshold_type.capitalize()}",
                         init_val=252, max_val=255)
        self.type = self.threshold_types[threshold_type]
        self.blur = self.img_processor.blur(self.gray)

    def process_images(self):
        _, self.simple_threshold = cv.threshold(
            self.blur, self.trackbar_value, self.max_val, self.type)
        return self.img_processor.process(self.simple_threshold)
