import cv2 as cv
import sys

# グレースケールで読み込み
img = cv.imread("sample/sheet2.png", cv.IMREAD_GRAYSCALE)
cv.imwrite("process1.png", img)

#ret, img_bin = cv.threshold(img, 130, 255, cv.THRESH_TOZERO)
# 白黒化
ret, img_bin = cv.threshold(img, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imwrite("process2.png", img_bin)

# 輪郭検出
cnt, hierarchy = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# 検出した輪郭を赤色で描画して保存
sheet = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
sheet = cv.drawContours(sheet, cnt, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
cv.imwrite("process3.png", sheet)

