import time
import cv2
import numpy as np  # them luc bat dau xu ly nhieu
from matplotlib import path  # them thu vien dem vat the trong vung chon

import myFunc as mf  # them luc bat dau ve hinh

cap = cv2.VideoCapture("road.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

bg = cv2.createBackgroundSubtractorMOG2()
wait_time = 1
while True:
    pre_time = time.time()
    ret,img = cap.read()

    bgmask = bg.apply(img)  # mat na chuyen dong

    bgmask = cv2.erode(bgmask, np.ones((10, 10)))  # loc nhieu
    bgmask = cv2.dilate(bgmask, np.ones((25,25)))

    if mf.getPath() is not None:  # tra ve nhung diem minh da ve, đỉnh của đa giác
        p = path.Path(np.array(mf.getPath()))  # bat dau goi ham
        pts = np.array([mf.getPath()])
        cv2.polylines(img, [pts],True, (255, 0, 0), 2)  # mau vung tron

        contours = cv2.findContours(bgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # tao vung khoang dem
        # CV_RETR_TREE : khi sử dụng cờ này nó lấy tất cả các đường biên và tạo ra một hệ thống phân cấp đầy đủ của những đường lồng nhau
        # CV_CHAIN_APPROX_SIMPLE : nó sẽ nén đường viền trước khi lưu trữ, nén phân đoạn theo chiều ngang, chiều dọc và chéo . Ví dụ : một hình chữ nhật sẽ được mã hoá bằng toạ độ của 4 đỉnh.
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        count = 0  # gan bien dem
        # if contours is not None:
        #     print len(contours)
        if len(contours) > 0:  # bat dau ve len, nếu tồn tại biên
            for c in contours:

                M = cv2.moments(c)
                x = int(M['m10'] / (M['m00'] + 1)) #toa do
                y = int(M['m01'] / (M['m00'] + 1))

                if p.contains_point((x, y)):
                    count = count + 1
                    cv2.putText(img, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255))  # in biến đếm lên ảnh ket thuc ve len

    bgmask = cv2.merge([bgmask, bgmask, bgmask])   #gộp mặt nạ lại 3 kênh
    result = cv2.bitwise_and(bgmask, img)  # ghép video gốc vs bgmask

    mf.selPic("Video", img)
    cv2.imshow("Video", img)
    cv2.imshow("bgmask", bgmask)
    cv2.imshow("result", result)

    delta_time = (time.time() - pre_time) * 1000  # Thời gian
    if delta_time > wait_time:
        delay_time = 1
    else:
        delay_time = wait_time - delta_time
    if cv2.waitKey(int(delay_time)) == ord('q'):
        break

cv2.destroyAllWindows()
