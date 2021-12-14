import cv2 as cv
from myocv.tb import Trackbar
import myocv.imageprocessing as impro


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
        self.gray = impro.gray(self.img)
        self.blur = impro.blur(self.gray)

    def process_images(self):
        _, self.simple_threshold = cv.threshold(
            self.blur, self.trackbar_value, self.max_val, self.type)
        return impro.draw_images(self.simple_threshold)


class CannyThreshold(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Canny") -> None:
        super().__init__(src_img_path, name, init_val=128, max_val=255)
        self.gray = impro.gray(self.img)
        self.blur = impro.blur(self.gray)

    def process_images(self):
        self.canny = cv.Canny(
            self.blur, self.trackbar_value, self.trackbar_value * 2)
        return impro.draw_images(self.canny)


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
        self.gray = impro.gray(self.img)
        self.blur = impro.blur(self.gray)

    def process_images(self):
        if self.trackbar_value % 2 == 0:
            self.trackbar_value += 1
        self.adapt = cv.adaptiveThreshold(self.blur, 255, self.type,
                                          cv.THRESH_BINARY, self.trackbar_value, 2)
        return impro.draw_images(self.adapt)


class OtsuThreshold(Trackbar):
    def __init__(self, src_img_path: str, name: str = "Threshold Otsu") -> None:
        super().__init__(src_img_path, name, init_val=0, max_val=255)
        self.gray = impro.gray(self.img)
        self.blur = impro.blur(self.gray)

    def process_images(self):
        _, self.otsu = cv.threshold(self.blur, self.init_val, self.max_val,
                                    cv.THRESH_BINARY+cv.THRESH_OTSU)
        return impro.draw_images(self.otsu)
