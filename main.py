import sys
import imghdr
from myocv.thresholds import SimpleThreshold
from myocv.thresholds import CannyThreshold
from myocv.thresholds import AdaptiveThreshold
from myocv.thresholds import OtsuThreshold
from myocv.tbwin import TrackbarWindow


def main():
    args = sys.argv[1:]

    if not len(args) or not imghdr.what(args[0]):
        print("Please provide an image")
        return

    image = args[0]

    simple = SimpleThreshold(image)
    canny = CannyThreshold(image)
    adapt = AdaptiveThreshold(image, threshold_type="gauss")
    otsu = OtsuThreshold(image)

    trackbars = [
        simple,
        canny,
        adapt
    ]

    window = TrackbarWindow(trackbars)
    window.show()


if __name__ == "__main__":
    main()
