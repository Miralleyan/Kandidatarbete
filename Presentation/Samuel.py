import numpy as np
from manim import *
import math
import torch
import copy
from manim.utils.color import Colors

class Samuel(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        CreateDots(self)
        self.wait(1)

def CreateDots(scene):
    X = 7
    Y = 7
    dots = []
    animation = []
    scale = 1.1
    for y in range(Y):
        row = []
        for x in range(X):
            dot = Dot(np.array([x*scale-4, -y*scale+3.5, 0]),radius=0.25, color=LIGHT_GREY)
            row.append(dot)
            animation.append(Create(dot))
        dots.append(row)
    
    time = 1
    lag = time/(X*Y)
    scene.play(Succession(
        *animation,
        lag_ratio=lag,
        run_time=time))
    # scene.play(*animation)
    # scene.play(Create(rect_1))
    # animations = []
    # animations.append(Transform(rect_1, rect_2))
    # time = 1.5
    # scene.lag_ratio = 0.1
    # scene.play(scene.h_dot.animate(run_time=time).shift(RIGHT*m*scene.y_value/8),
    #     Succession(
    #     *animations,
    #     lag_ratio=scene.lag_ratio,
    #     run_time=time),
    # )
def HighlightDots(scene):
    # for x in range(-7, 8):
    #         for y in range(-4, 5):
    #             scene.add(Dot(np.array([x, y, 0]), color=DARK_GREY))
    scene.play(Create(rect_1))
    animations = []
    animations.append(Transform(rect_1, rect_2))
    time = 1.5
    scene.lag_ratio = 0.1
    scene.play(scene.h_dot.animate(run_time=time).shift(RIGHT*m*scene.y_value/8),
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )