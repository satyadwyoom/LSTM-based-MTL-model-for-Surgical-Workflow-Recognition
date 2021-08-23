import os 
import cv2



##Main Code to iterate the directory and make frames for all videos
frames_path = '/arc/project/st-anaray02-1/skumar40/petraw_data/Training/Frames/'
vid_path = '/arc/project/st-anaray02-1/skumar40/petraw_data/Training/Video'

for file in os.listdir(vid_path):
        
    v_path = os.path.join(vid_path, file)
    
    
    if not os.path.isdir(frames_path + str(file)[:-4]):
        os.makedirs(frames_path + str(file)[:-4])
        os.chdir(frames_path + str(file)[:-4])
        print(str(file)[:-4])
        
    cap= cv2.VideoCapture(v_path)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(str(i)+'.jpg',frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()