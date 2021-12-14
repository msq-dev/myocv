import cv2 as cv

MIN_AREA = 70
MAX_AREA = 140


class ImageProcessor:
    @classmethod
    def process(cls, operation):
        contours = cls.find_contours(operation)
        return [
            cls.draw_contours(operation, contours),
            cls.draw_min_circle(operation, contours),
            cls.draw_bound_rect(operation, contours),
        ]

    @classmethod
    def draw_contours(cls, canvas, contours):
        if len(canvas.shape) < 3:
            canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        for c in contours:
            cv.convexHull(c)
            color = cls.correct_color(c)
            cv.drawContours(canvas, [c], -1, color, 2)
        return canvas

    @classmethod
    def draw_bound_rect(cls, canvas, contours):
        if len(canvas.shape) < 3:
            canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        cont_polygon = [None]*len(contours)
        bound_rect = [None]*len(contours)

        for i, c in enumerate(contours):
            cv.convexHull(c)
            cont_polygon[i] = cv.approxPolyDP(c, 3, True)
            bound_rect[i] = cv.boundingRect(cont_polygon[i])

        for i in range(len(contours)):
            color = cls.correct_color(contours[i])
            pt_01 = (int(bound_rect[i][0]), int(bound_rect[i][1]))
            pt_02 = (int(bound_rect[i][0]+bound_rect[i][2]),
                     int(bound_rect[i][1]+bound_rect[i][3]))
            cv.rectangle(canvas, pt_01, pt_02, color, 1)
        return canvas

    @classmethod
    def draw_min_circle(cls, canvas, contours):
        if len(canvas.shape) < 3:
            canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
        cont_polygon = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)

        for i, c in enumerate(contours):
            cv.convexHull(c)
            cont_polygon[i] = cv.approxPolyDP(c, 3, True)
            centers[i], radius[i] = cv.minEnclosingCircle(cont_polygon[i])

        for i in range(len(contours)):
            color = cls.correct_color(contours[i])
            cv.circle(canvas,
                      (int(centers[i][0]), int(centers[i][1])),
                      int(radius[i]), color, 2)
        return canvas

    @staticmethod
    def gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def blur(img):
        return cv.GaussianBlur(img, (5, 5), 1)

    @staticmethod
    def find_contours(img) -> tuple:
        contours, _ = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def correct_color(contour) -> tuple:
        color = (0, 255, 0)
        if MIN_AREA < cv.contourArea(contour) < MAX_AREA:
            color = (0, 0, 255)
        return color
