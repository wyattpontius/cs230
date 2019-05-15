import cv2
import PIL
from PIL import Image
import os
import imageio
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def process_video(clip_num):
    
    cap = cv2.VideoCapture('C:/Users/pablo/Videos/Captures/230Data_Clip' + str(clip_num) + '.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    
    root_path = 'C:/Users/pablo/Videos/230Frames/'
    i = 1
    while(cap.isOpened()):
            
        frame_exists, curr_frame = cap.read()
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        curr_path = root_path + 'clip' + str(clip_num + 1) + '/'
        if i%34 == 0:
            j = int (i/34)
            imageio.imwrite (curr_path + 'frame' + str(j + 13000) + '.jpg', curr_frame)
            if frame_exists:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
            else:
                break
        i += 1
    cap.release()
    
def main():
    first_clip = 74
    last_clip = 74
    pool = ThreadPool(last_clip - first_clip + 1)


    clip_nums = [i for i in range(first_clip, last_clip + 1)]
    pool.map(process_video, clip_nums)
    pool.close()
    pool.join()
    
main()

    


# =============================================================================
# for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
#     print('Frame %d difference:'%i, abs(ts - cts))
# =============================================================================
