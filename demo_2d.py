from datetime import time
import cv2
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from sys import argv
import pickle
import knn
import torch
import pyautogui
import time
from collections import Counter

"""
Read the movie located at moviePath, perform the 2d pose annotation and display
Run from terminal : python demo_2d.py [movie_file_path] [max_persons_detected]
with all parameters optional.
Keep holding the backspace key to speed the video 30x
"""

global label_pool
label_pool = ['stand_still','stand_still','stand_still','stand_still','stand_still']

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def keyUpAll(label):
    if label == "jump":
        pyautogui.keyUp('j')

    if label == "walk_forward":
        pyautogui.keyUp('w')
    if label == "walk_left":
        pyautogui.keyUp('a')
    if label == "walk_right":
        pyautogui.keyUp('d')

    if label == "run_forward":
        pyautogui.keyUp('w')
        pyautogui.keyUp('shift')
    if label == "run_left":
        pyautogui.keyUp('a')
        pyautogui.keyUp('shift')
    if label == "run_right":
        pyautogui.keyUp('d')
        pyautogui.keyUp('shift')

    if label == "creep_forward":
        pyautogui.keyUp('c')
        pyautogui.keyUp('w')
    if label == "creep_left":
        pyautogui.keyUp('c')
        pyautogui.keyUp('a')
    if label == "creep_right":
        pyautogui.keyUp('c')
        pyautogui.keyUp('d')

    if label == "dance":
        pyautogui.press('p')



def keyDownAll(label):
    if label == "walk_forward":
        pyautogui.keyDown('w')
    if label == "walk_left":
        pyautogui.keyDown('a')
    if label == "walk_right":
        pyautogui.keyDown('d')

    if label == "run_forward":
        pyautogui.keyDown('w')
        pyautogui.keyDown('shift')
    if label == "run_left":
        pyautogui.keyDown('a')
        pyautogui.keyDown('shift')
    if label == "run_right":
        pyautogui.keyDown('d')
        pyautogui.keyDown('shift')

    if label == "creep_forward":
        pyautogui.keyDown('c')
        pyautogui.keyDown('w')
    if label == "creep_left":
        pyautogui.keyDown('c')
        pyautogui.keyDown('a')
    if label == "creep_right":
        pyautogui.keyDown('c')
        pyautogui.keyDown('d')

    if label == "dance":
        pyautogui.press('p')
        
def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num
 
def keyboard_control(label):
    '''
    label_names = ['creep_forward', 'creep_left', 'creep_right', \
               'dance', 'jump', 'punch', 'run_forward', \
               'run_left', 'run_right', 'stand_still',  \
               'walk_forward', 'walk_left', 'walk_right']
    '''
    # get last control label
    label_count = Counter(label_pool[-5:])
    old_label, _ = label_count.most_common()[0]
    #  get new label pool
    label_pool.append(label)
    # get current control label
    label_count = Counter(label_pool[-5:])
    new_label, _ = label_count.most_common()[0]

    # keep old keyDown
    if (old_label == new_label): return
    # jump don't need to keyUp
    if old_label == "jump":
        pyautogui.keyUp('j')
    # jump don't need to keyUp
    if new_label == "jump":
        pyautogui.keyDown('j')
        return

    # find new key input, reset all the other keys
    keyUpAll(old_label)
    # set new input
    keyDownAll(new_label)



def start(movie_path, max_persons):

    annotator = AnnotatorInterface.build(max_persons=max_persons)

    cap = cv2.VideoCapture(movie_path)

    joint_pos = []
    ignore_num = 100
    count = 0 
    class_name = "test" 
    # all classes: "walk_forward" "walk_left" "walk_right"
    #              "run_forward" "run_left" "run_right" 
    #              "creep_forward" "creep_left" "creep_right"
    #              "jump" "dance" "punch" "stand_still"
    frame_counter = 0
    while(True):

        ret, frame = cap.read()

        if not ret:
            break
        # if (count >= ignore_num) and (len(persons) > 0):
        #     cv2.imwrite(f"./pose_dataset/without_pose/{frame_counter:06d}.jpg", frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tmpTime = time.time()
        persons = annotator.update(frame)

        # store pose information
        if (count >= ignore_num) and (len(persons) > 0):
            curr_pos = persons[0]['pose_2d'].joints
            # body_center = curr_pos.mean(axis=0)
            # joint_pos.append(persons[0]['pose_2d'].joints.reshape(-1))
            joint_pos.append(knn.normalized_frame(curr_pos.reshape(-1)))
            if len(joint_pos) > 10:
                predictions = knn.knn.predict(torch.tensor(joint_pos[-10:]).flatten().unsqueeze(0))
                label = knn.label_names[predictions]
                print(label, "\n")
                keyboard_control(label)
        else:
            count += 1

        fps = int(1/(time.time()-tmpTime+0.0001))

        poses = [p['pose_2d'] for p in persons]

        ids = [p['id'] for p in persons]
        frame = Drawer.draw_scene(frame, poses, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if (count >= ignore_num) and (len(persons) > 0):
        #     cv2.imwrite(f"./pose_dataset/with_pose/{frame_counter:06d}.jpg", frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(33) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

        frame_counter += 1

    annotator.terminate()
    cap.release()
    cv2.destroyAllWindows()


    # joint_pos = joint_pos[:-ignore_num]
    # save(f"./pose_dataset/{class_name}.pickle", joint_pos)




if __name__ == "__main__":

    print("start frontend")

    default_media = 0
    max_persons = 2

    if len(argv) == 3:
        default_media = 0 if argv[1] == "webcam" else argv[1]
        start(default_media, int(argv[2]))
    elif len(argv) == 2:
        default_media = 0 if argv[1] == "webcam" else argv[1]
        start(default_media, max_persons)
    else:
        start(default_media, max_persons)



