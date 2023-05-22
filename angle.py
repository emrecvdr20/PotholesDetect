import cv2
import numpy as np
import math

def distance(x1, y1, x2, y2):
    dist = math.sqrt(math.fabs(x2-x1)**2 + math.fabs(y2-y1)**2)
    return dist

def find_orange(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_lowerbound =  np.array([15,154,0])
    hsv_upperbound = np.array([32,215,255])
    
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask) 
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 1000:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy), True
        else:
            return (700, 700), False 
    else:
        return (700, 700), False 

def find_black(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_lowerbound = np.array([41,21,0]) 
    hsv_upperbound = np.array([107,88,42])
    mask = cv2.inRange(hsv_frame, hsv_lowerbound, hsv_upperbound)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cnts, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        maxcontour = max(cnts, key=cv2.contourArea)

        M = cv2.moments(maxcontour)
        if M['m00'] > 0 and cv2.contourArea(maxcontour) > 2000:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy), True 
        else:
            return (700, 700), True 
    else:
        return (700, 700), True

cap = cv2.VideoCapture(0)

while(1):
    _, orig_frame = cap.read()
    print(orig_frame)
    copy_frame = orig_frame.copy() 
    (color1_x, color1_y), found_color1 = find_orange(copy_frame)
    (color2_x, color2_y), found_color2 = find_black(copy_frame)

    cv2.circle(copy_frame, (color1_x, color1_y), 20, (255, 0, 0), -1)
    cv2.circle(copy_frame, (color2_x, color2_y), 20, (0, 128, 255), -1)

    if found_color1 and found_color2:

        hypotenuse = distance(color1_x, color1_x, color2_x, color2_y)
        horizontal = distance(color1_x, color1_y, color2_x, color1_y)
        vertical = distance(color2_x, color2_y, color2_x, color1_y)
        angle = np.arcsin(vertical/hypotenuse)*180.0/math.pi

        cv2.line(copy_frame, (color1_x, color1_y), (color2_x, color2_y), (0, 0, 255), 2)
        cv2.line(copy_frame, (color1_x, color1_y), (color2_x, color1_y), (0, 0, 255), 2)
        cv2.line(copy_frame, (color2_x, color2_y), (color2_x, color1_y), (0, 0, 255), 2)

        angle_text = ""
        if color2_y < color1_y and color2_x > color1_x:
            angle_text = str(int(angle))
            """Eğer ikinci renk noktasi, birinci renk noktasindan daha yukarida ve daha sağda ise,
              açiyi hesapla ve doğrudan aciyi "angle_text" değişkenine kaydet."""
        elif color2_y < color1_y and color2_x < color1_x:
            angle_text = str(int(180 - angle))
            """Eğer ikinci renk noktasi, birinci renk noktasindan daha yukarida ve daha solda ise,
              açiyi hesapla ve 180 derece ile açiyi çikararak "angle_text" değişkenine kaydet."""
        elif color2_y > color1_y and color2_x < color1_x:
            angle_text = str(int(180 + angle))
            """Eğer ikinci renk noktasi, birinci renk noktasindan daha aşağida ve daha solda ise,
              açiyi hesapla ve 180 derece ile açiyi toplayarak "angle_text" değişkenine kaydet."""
        elif color2_y > color1_y and color2_x > color1_x:
            angle_text = str(int(360 - angle))
            """Eğer ikinci renk noktasi, birinci renk noktasindan daha aşagida ve daha sağda ise,
              açiyi hesapla ve 360 derece ile açiyi çikararak "angle_text" değişkenine kaydet."""
 
        cv2.putText(copy_frame, angle_text, (color1_x-30, color1_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 128, 229), 2)

    cv2.imshow('AngleCalc', copy_frame)
    cv2.waitKey(1) 

cap.release()
cv2.destroyAllWindows()