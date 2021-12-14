import cv2 as cv
from myocv.tb import Trackbar


class AdaptiveThreshold(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Adaptive",
                 threshold_type: str = "mean") -> None:
        super().__init__(src_img_path,
                         name=f"{name} {threshold_type.capitalize()} C",
                         init_val=3, max_val=555)
        self.threshold_types = {
            "mean": cv.ADAPTIVE_THRESH_MEAN_C,
            "gauss": cv.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        self.type = self.threshold_types[threshold_type]
        self.blur = self.img_processor.blur(self.gray)

    def process_images(self):
        if self.trackbar_value % 2 == 0:
            self.trackbar_value += 1
        self.adapt = cv.adaptiveThreshold(self.blur, 255, self.type,
                                          cv.THRESH_BINARY, self.trackbar_value, 2)
        return self.img_processor.process(self.adapt)
