import cv2 as cv


class Trackbar:
    def __init__(self,
                 name: str,
                 init_val: int,
                 max_val: int) -> None:

        self.name = name
        self.init_val = init_val
        self.max_val = max_val
        self.trackbar_value = init_val

    def get_position(self, win_name: str) -> None:
        self.trackbar_value = cv.getTrackbarPos(self.name, win_name)

    def process_images(self) -> list:
        pass

    def create(self, win_name: str, callback) -> None:
        cv.createTrackbar(self.name,
                          win_name,
                          self.init_val,
                          self.max_val,
                          callback)
