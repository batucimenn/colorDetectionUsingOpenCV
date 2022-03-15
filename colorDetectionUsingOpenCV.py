import cv2
import numpy as np
from collections import deque

buffer_size = 16
pts = deque(maxlen = buffer_size)

# istediğimiz renk aralıklarını hsv formatında belirtiyoruz
blackLower = (0, 0, 0)
blackUpper = (50, 50, 50)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success, imgOriginal =  cap.read()
    if success:
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0) # blurlama
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # görüntüyü bgr formatından hsv formatına çeviriyoruz
        cv2.imshow("HSV Image", hsv)
        mask = cv2.inRange(hsv, blackLower, blackUpper) # oluşturduğumuz renk aralıklarında kalan rengi maskeliyoruz
        cv2.imshow("Mask Image", mask)
        # görüntümüzdeki gürültüleri azaltmak için erode ve dilate fonksiyonlarını kullanıyoruz
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)
        cv2.imshow("Erode + Dilate Image", mask)
        # maskelediğimiz rengin köşelerini buluyoruz
        (countours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if len(countours) > 0:
            # maskelediğimiz görüntümüzde en büyük alana sahip istediğimiz rengin noktalarını alıyoruz
            c = max(countours, key = cv2.contourArea)
            # dikdörtgene çevir
            rect = cv2.minAreaRect(c)
            ((x, y), (width, height), rotation) = rect
            s = f"x: {np.round(x)}, y: {np.round(y)}, width: {np.round(width)}, height: {np.round(height)}, rotation: {np.round(rotation)}"
            print(s)
            # kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            # moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            # konturu çizdir: sarı
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255), 2)
            # merkeze bir tane nokta çizelim
            cv2.circle(imgOriginal, center, 3, (255,0,255), -1)
            # bilgileri ekrana yazdır
            cv2.putText(imgOriginal, s, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2) 
            cv2.imshow("Orijinal Tespit", imgOriginal) 
    if cv2.waitKey(1) & 0xFF == ord("q"): break