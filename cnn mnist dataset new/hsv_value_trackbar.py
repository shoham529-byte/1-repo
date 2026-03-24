import cv2
import numpy as np

def empty(a):
 
    pass

def main():
 
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    
   )
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

 
    cap = cv2.VideoCapture(0)

    print("--- HSV Calibration Tool ---")
    print("1. Hold your pen up to the camera.")
    print("2. Adjust the sliders until ONLY your pen is white in the 'Mask' window.")
    print("3. Press 's' to SAVE the values to hsv_value.npy and quit.")
    print("4. Press 'q' to QUIT without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
      
        frame = cv2.flip(frame, 1)
        

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])

    
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask (Aim for white pen, black background)", mask)
        cv2.imshow("Result (Isolated Color)", result)

       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting without saving.")
            break
        elif key == ord('s'):
        
            np.save('hsv_value.npy', np.array([lower_bound, upper_bound]))
            print(f"Saved! Lower: {lower_bound}, Upper: {upper_bound}")
            print("You can now run live_writing.py!")
            break

   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()