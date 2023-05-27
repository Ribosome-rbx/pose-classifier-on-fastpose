from datetime import time
import cv2
from src.system.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from sys import argv
import pickle
import knn
import torch

"""
Read the movie located at moviePath, perform the 2d pose annotation and display
Run from terminal : python demo_2d.py [movie_file_path] [max_persons_detected]
with all parameters optional.
Keep holding the backspace key to speed the video 30x
"""

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

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



