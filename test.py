import imUtils
import configure as cfg
import cv2

folder = 'test_images/'
img = imUtils.imread(folder + '1.cr3')
gray = cv2.cvtColor(imUtils.toOpenCVU8(img.copy()), cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,27,9)
# apply close morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
imUtils.imshow(thresh, 'thresh')

(cnts, _) = cv2.findContours(thresh.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imUtils.drawCnts(img.copy(), cnts)
