import cv2
import os
import imageio as imio




start = 500
fps = 40
end = 1500
pth = "data/trials/angle_vis/"
# video_name = "demopushervid.mp4"
# pth = "data/goodgripper/0/"
top_pth = "data/goodgripper/"
video_name = "angle_vis.avi"
# pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# top_pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# video_name = "ballintensity.mp4"

prefix = "param_frame"
# prefix = "intensity_"
# prefix = "marker_"
# prefix = "focus_img_"


class dummyargs():
    def __init__(self):
        self.record_rollouts = top_pth
        self.num_iters = -1

args = dummyargs()

im = cv2.imread(pth + prefix + str(start) + ".png")
print(im, pth + prefix + str(start) + ".png")
height, width, layers = im.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))
for i in range(start,end):
    im = cv2.imread(pth + prefix + str(i) + ".png")
    if im is not None:
        video.write(im)
        cv2.imshow('frame',im)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
video.release()
