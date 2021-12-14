import sys
import imghdr
from myocv.canny import Canny
from myocv.simpth import SimpleThreshold
from myocv.adaptith import AdaptiveThreshold
from myocv.otsuth import OtsuThreshold
from myocv.tbwin import TrackbarWindow


def main():
    args = sys.argv[1:]

    if not len(args) or not imghdr.what(args[0]):
        print("Please provide an image")
        return

    image = args[0]
    # image_02 = args[1]

    simple = SimpleThreshold(image)
    canny = Canny(image)
    adapt = AdaptiveThreshold(image, threshold_type="gauss")
    # simple_02 = SimpleThreshold(image, threshold_type="tozero")
    # canny_02 = Canny(image_02, "Canny 2")
    # otsu = OtsuThreshold(image)

    trackbars = [
        simple,
        canny,
        adapt
        # simple_02,
        # canny_02
        # otsu,
    ]

    window = TrackbarWindow(trackbars)
    window.show()


if __name__ == "__main__":
    main()
