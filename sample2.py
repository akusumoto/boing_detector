import cv2 as cv
import numpy

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
sheet3 = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
sheet3 = cv.drawContours(sheet3, cnt, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
cv.imwrite("process3.png", sheet3)

sheet5 = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
sheet5_onlyline = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
sheet5_bars = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)

height, width = img.shape
image_size = height * width
filtered_cnt = []
line_contour_boundings = [] # 五線自体
for i, contour in enumerate(cnt):
    area = cv.contourArea(contour)
    if image_size * 0.99 < area:
        continue
    filtered_cnt.append(contour)

    x,y,w,h = cv.boundingRect(contour)
    sheet5 = cv.rectangle(sheet5, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # 紙の幅の6割を超えるエリアのものは五線を検出したものとみなす
    if w > width * 0.6:
        line_contour_boundings.append((x,y,w,h))
        sheet5_onlyline = cv.rectangle(sheet5_onlyline, (x, y), (x+w, y+h), (0, 255, 0), 3)
        sheet5_bars = cv.rectangle(sheet5_bars, (x, y), (x+w, y+h), (0, 255, 0), 3)


sheet4 = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
sheet4 = cv.drawContours(sheet4, filtered_cnt, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
cv.imwrite("process4.png", sheet4)

cv.imwrite("process5.png", sheet5)
cv.imwrite("process5_1.onlyeline.png", sheet5_onlyline)

bar_x_candidates = {}
x_range = 3 #px ?
for contour in filtered_cnt:
    # line no: 1.. (y, h)
    # bar catidate x position: x
    # detection count: 0..
    #   (line_y, line_height, bar_cx) = detect_count
    x,y,w,h = cv.boundingRect(contour)
    for line_x, line_y, line_w, line_h in line_contour_boundings:
        if line_x <= x and x <= line_x + line_w and \
           line_y <= y and y <= line_y + line_h and \
           w > h * 5: # 五線の間を検出しているものは横長のはず
            # 五線エリア内
            for (c_line_x, c_line_w, c_line_y, c_line_h, c_x), count in list(bar_x_candidates.items()):
                if (c_x - x_range) <= (x + w) and (x + w) <= (c_x + x_range):
                    # 同じ x の箇所（見つけたエリアの右側 (x+w)）（誤差 x_range 内）を見つけたらカウントアップしておく
                    bar_x_candidates[(c_line_x, c_line_w, c_line_y, c_line_h, c_x)] = count + 1
                    break
            else: 
                bar_x_candidates[(line_x, line_w, line_y, line_h, (x + w))] = 1

for (line_x, line_w, line_y, line_h, x), count in list(bar_x_candidates.items()):
    if count <= 4:
        sheet5_bars = cv.line(sheet5_bars, (x, line_y), (x, line_y + line_h), (0, 255, 0), 3)
cv.imwrite("process5_2.bars.png", sheet5_bars)


# 線を検出
lines = cv.HoughLinesP(img_bin, rho=1, theta=numpy.pi/360, threshold=120, minLineLength=1000, maxLineGap=100)
sheet6 = cv.imread("sample/sheet2.png", cv.IMREAD_COLOR)
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 < width*0.01 or x1 > width*0.99 or x2 < width*0.01 or x2 > width*0.99 or \
       y1 < height*0.01 or y1 > height*0.99 or y2 < height*0.01 or y2 > height*0.99:
       continue
    sheet6 = cv.line(sheet6, (x1, y1), (x2, y2), (0, 0, 255), 1)
cv.imwrite("process6.png", sheet6)