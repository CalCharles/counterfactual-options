import cv2
import os
import imageio as imio
import sys



start = 1
fps = 40
end = 2000
# pth = "data/trials/angle_vis/"
# pth = "data/breakout/test_big_block/3/"
# pth = "data/robopushing/path_test/0/"
# video_name = "demopushervid.mp4"
# pth = "data/goodgripper/0/"
top_pth = "/hdd/datasets/counterfactual_data/breakout/videos/big_block"
video_name = "big_block.avi"
# pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# top_pth = "results/cmaes_soln/focus_self/focus_img_ball1_25/intensity/"
# video_name = "ballintensity.mp4"

prefix = "state"
# prefix = "intensity_"
# prefix = "marker_"
# prefix = "focus_img_"
if __name__ == "__main__":


    class dummyargs():
        def __init__(self):
            self.record_rollouts = top_pth
            self.num_iters = -1

    args = dummyargs()

    pth = sys.argv[1]
    prefix = sys.argv[2]
    video_name = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    im = cv2.imread(pth + prefix + str(start) + ".png")
    print(im, pth + prefix + str(start) + ".png")
    height, width, layers = im.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))
    for i in range(start,end):
        im = cv2.imread(pth + prefix + str(i) + ".png")
        if im is not None:
            video.write(im)
            # cv2.imshow('frame1',im)
            # if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            #     break
    cv2.destroyAllWindows()
    video.release()
