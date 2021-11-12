import cv2
import time

#storing input video file + model files
input_file = 'test.mp4'
video_cap = cv2.VideoCapture(input_file)
model = 'cars.xml'

#using pre-defined model
carModel = cv2.CascadeClassifier(model)

start = time.time()

while True:
    
    # starting to read video frames
    (run, frame) = video_cap.read()

    if run:
        #convert to grey scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rescaling frame
        width = int(frame.shape[1] * .9)
        height = int(frame.shape[0] * .9)
        frame = cv2.resize(frame, (width,height))

    #detecting cars
    car_objects = carModel.detectMultiScale(gray_frame,1.1,7)
    end = time.time()
    #drawing boxes around cars
    for (x, y, w, h) in car_objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'CAR', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # displays GUI with running the video frame
    cv2.imshow('Detecting cars on the street...',frame)
    
    # printing detection coordinates 
    print("Cars detected at: " + "[" + str(x) + ", " + str(y) + "]")
    
    if(cv2.waitKey(1) == 27):
        break
    
video_cap.release()
print("Duration Time: " + str(end-start) + " seconds")


