import cv2

def main():
    cap = cv2.VideoCapture("rtsp://172.22.48.1:8554/webcam.h264") 
    print("a")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()