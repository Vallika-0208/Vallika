import cv2
import numpy as np

# Edge Detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Region of Interest
def region_of_interest(image):
    height = image.shape[0]

    polygons = np.array([
        [(100,height),
         (550,250),
         (900,height)]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    cropped = cv2.bitwise_and(image, mask)

    return cropped

# Draw Lines
def display_lines(image, lines):

    line_image = np.zeros_like(image)

    if lines is not None:

        for line in lines:

            x1,y1,x2,y2 = line.reshape(4)

            cv2.line(
                line_image,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                5
            )

    return line_image

cap = cv2.VideoCapture(r"D:\AI Self_car\test_video.mp4.mp4")

if not cap.isOpened():
    print("Video open kaaledu!")
    exit()

print("Video opened successfully!")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    canny_image = canny(frame)

    cropped = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi/180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    line_image = display_lines(frame, lines)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Lane Detection", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
