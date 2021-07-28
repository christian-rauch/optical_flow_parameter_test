#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import time


class FlowVis:
    def __init__(self, input1, input2) -> None:

        self.im1 = cv2.cvtColor(cv2.imread(input1), cv2.COLOR_BGR2GRAY)
        self.im2 = cv2.cvtColor(cv2.imread(input2), cv2.COLOR_BGR2GRAY)

        # set up UI elements
        self.win_name = "flow"
        cv2.namedWindow(self.win_name)

        # slider for Farnebäck optical flow parameters
        self.scale0 = 1.0
        self.scale = 0.5
        self.lvls = 3
        self.winsize = 15
        self.iter = 3
        self.polyn = 5
        self.sigma = 1.2
        cv2.createTrackbar('downscale (1e-2)', self.win_name, int(self.scale0 * 1e2), 100, self.on_downscale)
        cv2.createTrackbar('scale (1e-2)', self.win_name, int(self.scale * 1e2), 100, self.on_scale)
        cv2.createTrackbar('levels', self.win_name, self.lvls, 10, self.on_levels)
        cv2.createTrackbar('winsize', self.win_name, self.winsize, 100, self.on_winsize)
        cv2.createTrackbar('iterations', self.win_name, self.iter, 20, self.on_iterations)
        cv2.createTrackbar('neighbourhood', self.win_name, self.polyn, 20, self.on_polyn)
        cv2.createTrackbar('sigma (1e-1)', self.win_name, int(self.sigma * 1e1), 50, self.on_sigma)

        self.compute_flow()

    def on_downscale(self, scale):
        self.scale0 = max(1, scale) * 1e-2
        self.compute_flow()

    def on_scale(self, scale):
        self.scale = min(scale, 99) * 1e-2
        self.compute_flow()

    def on_levels(self, lvls):
        self.lvls = lvls
        self.compute_flow()

    def on_winsize(self, winsize):
        self.winsize = winsize
        self.compute_flow()

    def on_iterations(self, iter):
        self.iter = iter
        self.compute_flow()

    def on_polyn(self, polyn):
        self.polyn = polyn
        self.compute_flow()

    def on_sigma(self, sigma):
        self.sigma = sigma * 1e-1
        self.compute_flow()

    def compute_flow(self):
        # downscale for better performance
        im1s = cv2.resize(self.im1, None, fx = self.scale0, fy = self.scale0)
        im2s = cv2.resize(self.im2, None, fx = self.scale0, fy = self.scale0)

        tstart = time.time()
        flow = cv2.calcOpticalFlowFarneback(im1s, im2s, None, self.scale, self.lvls, self.winsize, self.iter, self.polyn, self.sigma, 0)
        duration = time.time() - tstart

        flow_s = cv2.resize(flow, None, fx = 1/ self.scale0, fy = 1/self.scale0)

        flow_s = cv2.resize(flow, (640, 480))

        mag, ang = cv2.cartToPolar(flow_s[...,0], flow_s[...,1])
        hsv = np.zeros((flow_s.shape[0], flow_s.shape[1], 3), np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        message = "scale: {:.2f}, level: {}, winsize: {}, iter {}, neighb.: {}, sigma {:.1f}".format(self.scale, self.lvls, self.winsize, self.iter, self.polyn, self.sigma)
        cv2.putText(flow_bgr, message, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = (255, 255, 255))
        cv2.putText(flow_bgr, "duration: {:.3f} s".format(duration), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = (255, 255, 255))
        cv2.imshow(self.win_name, flow_bgr)

    def run(self):
        # wait for ESC key or closed window
        # NOTE: Checking if a window is visible via 'getWindowProperty' requires the Qt backend.
        #       Install a Qt-enabled version via pip: 'pip3 install opencv-python'
        while ((cv2.waitKey(30) & 0xff) != 27) and \
              (cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) == 1.0):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input1')
    parser.add_argument('input2')
    args = parser.parse_args()

    FlowVis(args.input1, args.input2).run()
