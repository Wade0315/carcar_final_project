import cv2
import numpy as np

lower_white = np.array([0, 180, 0])
upper_white = np.array([180, 255, 40])

def detail_single_alternation(dir, image_file):
    image = cv2.imread(f"{dir}/{image_file}")
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    cv2.imshow(f"{image_file}", white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# def statistic_result(dir):
#     while True:
#         image = cv2.imshow(f"")
    

if __name__ == "__main__":
    detail_single_alternation("stock", "test_image.jpg")