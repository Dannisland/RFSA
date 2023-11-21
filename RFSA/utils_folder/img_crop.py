import cv2

# 裁剪
# for i in range(1,5):
#     path1 = "fig3_jpegvs"
#     path2 = "fig3"
#     s1 = "ours"
#     s2 = ""
#     img = cv2.imread(r"C:/Users/Dannis/Desktop/" + path1 + "/" + s1 + str(i) + ".png")
#     # print(img.shape)
#     cropped = img[1:257, 1:257]  # 裁剪坐标为[y0:y1, x0:x1]
#     cv2.imwrite(r"C:/Users/Dannis/Desktop/resize/" + path2 + "/" + s1 + str(i) + ".png", cropped)


# 画框
# for i in range(1,7):
#     path1 = "fig3_jpegvs"
#     path2 = "fig3"
#     s1 = ["abdhs","s","udhs","dahs","jaiss","ours"]
#     s2 = "4"
#
#     img = cv2.imread(r"C:/Users/Dannis/Desktop/fig1/" + s1[i-1] + s2 + ".png")
#
#     h = 60
#     h2 = h+1
#     x_center = 150
#     y_center = 125
#
#     crop = img[y_center-h2:y_center+h2,x_center-h2: x_center + h2]
#     cv2.imwrite(r"C:/Users/Dannis/Desktop/fig1_crop/" + s1[i-1] + s2 + ".png", crop)
#
#     draw_0 = cv2.rectangle(img, (x_center - h, y_center - h), (x_center + h, y_center + h), (0, 0, 255), 2)
#     cv2.imwrite(r"C:/Users/Dannis/Desktop/fig1_tangle/" + s1[i-1] + s2 + ".png", draw_0)
#     # cv2.imshow("1",draw_0)
#     # cv2.waitKey(0)

# 残差
for i in range(1,6):
    s1 = ["abdhs","udhs","dahs","jaiss","ours"]
    s2 = "4"

    img1 = cv2.imread(r"C:/Users/Dannis/Desktop/fig1_crop/s" + s2 + ".png")
    img2 = cv2.imread(r"C:/Users/Dannis/Desktop/fig1_crop/" + s1[i-1] + s2 + ".png")

    # img2 = cv2.imread(r"C:/Users/Dannis/Desktop/fig1_crop/jais1.png")
    res = cv2.absdiff(img1,img2)
    # res = img1-img2
    cv2.imwrite(r"C:/Users/Dannis/Desktop/fig1_res/" + s1[i-1] + s2 + ".png", res*10)

# img1 = cv2.imread(r"C:/Users/Dannis/Desktop/fig1_crop/c2.png")
# img2 = cv2.imread(r"C:/Users/Dannis/Desktop/fig1_crop/our2.png")
# res = cv2.absdiff(img1, img2)
# # cv2.imshow("1",res*2)
# # cv2.waitKey(0)
# cv2.imwrite(r"C:/Users/Dannis/Desktop/fig1_res/2.png", res*2)

