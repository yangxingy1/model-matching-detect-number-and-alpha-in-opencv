import cv2
import numpy as np
import myutils


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 模板
img_mod = cv2.imread("mod_letter.png")
# 待测
img_text = cv2.imread("test_letter.jpg")

# 模板处理
img_mod_gray = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(img_mod_gray, 147, 255, cv2.THRESH_BINARY_INV)[1]

refCnt, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_mod, refCnt, -1, (0, 0, 255), 3)
# 模板轮廓排序，从左到右，从上到下
refCnt = myutils.sort_contours(refCnt, method="left-to-right")[0]
digits = {}

# 重新定义模板大小
for (i, c) in enumerate(refCnt):
	(x, y, w, h) = cv2.boundingRect(c)
	roi = img_mod[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))
		# 每一个数字对应每一个模板
	digits[i] = roi
# 图片处理
img_text_gray = cv2.cvtColor(img_text, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_text_gray, 127, 255, cv2.THRESH_BINARY)
# 开运算（腐蚀后膨胀）
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh.copy(),cv2.MORPH_OPEN,kernel)
# 获得轮廓
contours_1,hierarchy = cv2.findContours(opening.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_text, contours_1, -1, (0, 255, 0), 3)
# cv2.namedWindow("img")
# cv2.moveWindow("img", 1000, 100)
cv_show("img",img_text)
# 挑选数字轮廓
lst = []
for (i,c) in enumerate(contours_1):
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# if 0 < ar < 100:
	# 	if (20 < w < 100) and (10 < h < 100):
	lst.append(c)
# 重排数字轮廓
lst = myutils.sort_contours(lst, method="lift-to-right,up-to-down")[0]
lst1 = []
final_result = []
# 重新定义对象大小
for c in lst:
	(x, y, w, h) = cv2.boundingRect(c)
	lst1.append((x,y,w,h))
# text2
# 	if y < 84:
# text1
# 	if y != 114:

# text3
	#if 48 < h < 100 and w > 45:
	roi_2 = img_text[y:y + h, x:x + w]
	roi_2 = cv2.resize(roi_2, (57,88))
	lst1.append((x,y,w,h))

	cv_show("img",roi_2)
	scores = [0]
# 计算匹配得分
	for (digit,digit_ROI) in digits.items():
		result = cv2.matchTemplate(roi_2,digit_ROI,cv2.TM_CCOEFF)
		(_,score,_,_) = cv2.minMaxLoc(result)
		scores.append(score)
	final_result.append(scores.index(max(scores)))
# 结果显示
dic_result = {1:"A",2:"B",3:"C"}

for i in final_result:
	print(f"{dic_result[i]}",end="")

# for (m,n,p,q) in lst1:
# 	print(m,n,p,q)
# 更改窗口位置
# cv2.moveWindow("img", 1000, 100)

