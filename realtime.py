from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time



model = load_model('E:/Final_Project_AI/Facial_landmark_upgrade_3.h5')  # <-- Saved model path


# input video file path
input_file = 'E:/Project_giuaky_thigiacmay/NhanDangKhuonMat/video/Toi.mp4'


# output file path
output_filename = 'E:/Final_Project_AI/testVideo_out.avi'  


def get_points_main(img):

    def detect_points(face_img):
        me  = np.array(face_img)/255
        
        x_test = np.expand_dims(me, axis=0)
        
        x_test = np.expand_dims(x_test, axis=3)
         
        y_test = model.predict(x_test)
        
        #after this instruction,variable 'label_points' have 30 values.
        label_points = np.squeeze(y_test)

        


        return label_points

    # load haarcascade
    face_cascade = cv2.CascadeClassifier('E:/Final_Project_AI/Step1_collect_data/haarcascade_frontalface_default.xml')
    dimensions = (96, 96)


    try:
        
        #cv2 return the BGR value, therefore we have to convert it to RGB
        default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        gray_img = cv2.cvtColor(default_img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

#         faces = face_cascade.detectMultiScale(gray_img, 4, 6)

    except:
        return []

    faces_img = np.copy(gray_img)
    
    plt.rcParams["axes.grid"] = False


    all_x_cords = []
    all_y_cords = []


    for i, (x,y,w,h) in enumerate(faces):
    

        try:
            just_face = cv2.resize(gray_img[y:y+h,x:x+w], dimensions)
        except:
            return []
        cv2.rectangle(faces_img,(x,y),(x+w,y+h),(255,0,0),-1)

        scale_val_x = w/96
        scale_val_y = h/96

        label_point = detect_points(just_face)
        
        all_x_cords.append((label_point[::2]*scale_val_x)+x)
        all_y_cords.append((label_point[1::2]*scale_val_y)+y)



    final_points_list = []
    try:
        for ii in range(len(all_x_cords)):
           for a_x, a_y in zip(all_x_cords[ii], all_y_cords[ii]):
                final_points_list.append([a_x, a_y])
    except:
        return final_points_list

    return final_points_list

cap = cv2.VideoCapture(0)


#cap = cv2.VideoCapture(input_file)
success, frame = cap.read()
height, width, channel = frame.shape
#height=720,width=1280 ,channel=3
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_filename, fourcc, 29, (width, height))


frame_no = 0
while cap.isOpened():

    a = time.time()
    frame_no += 1
    success, frame = cap.read()
    ch = cv2.waitKey(13) 
    if frame_no >10000:
        print("Exceeded esstential frame numbers")
        break
    
    if frame_no in range(1, 10000):
        points = get_points_main(frame)
        #print(points)
        try:
            
            overlay = frame.copy()
        except Exception as e:
            print(e)
            break

        for point in points:
            x=round(point[0])
            y=round(point[1])
            #print(x,y)
            cv2.circle(frame, (x,y), 3, (255, 255, 255),5)

        if len(points) != 0:
            o_line_points = [[12,13], [13,11], [11,14], [14,12], [12,10], [11,10], [10,3], [12,5], [11,3], [10,5], [10,4], [10,2], [5,1], [1,4], [2,0], [0,3], [5,9], [9,8], [8,4], [2,6], [6,7], [7,3]]
            num_face = len(points)//15

            for i in range(num_face):
                line_points = np.array(o_line_points) + (15*(i))

                the_color = (124,252,0)

                for ii in line_points:
                    points[ii[0]]=np.array(points[ii[0]])
                    points[ii[1]]=np.array(points[ii[1]])
                    x1_initial=round(points[ii[0]][0])
                    y1_initial=round(points[ii[0]][1])
                    x1_final=round(points[ii[1]][0])
                    y1_final=round(points[ii[1]][1])
              
                    cv2.line(overlay , (x1_initial,y1_initial),(x1_final,y1_final) ,the_color, thickness=2)


        opacity = 0.3
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        b = time.time()
        
        out.write(frame)
        
        cv2.imshow('frame',frame)
      
        
       

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
           

cap.release()
cv2.destroyAllWindows()
