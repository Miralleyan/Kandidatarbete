
import math


import numpy as np
from manim import *
import math
import torch
import copy
from manim.utils.color import Colors

scale= 6
N = 4

class Softmax(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        self.start_values = [0.2, 0.1, 0.3, 0.4]
        self.m = sum(self.start_values)
        self.chart = create_barchart(self, [0]*scale*2)
        HistogramRepresentation(self)
        # self.wait(2)
        DisplayTotalMass(self)
        # self.wait(1)
        AddMass(self)
        # self.wait(1)
        Softmax_scene(self)
        # self.wait(1)
        show_both(self)
        self.wait(1)
    
def create_barchart(scene, values):
    chart = BarChart(
        values=values,
        bar_names=[""]*scale*2,
        y_range=[0, 1, 1/10],
        x_length=scale,
        y_length=6,
        bar_width=0.5, 
        bar_fill_opacity=0.5,
        bar_colors = [BLACK],
        fill_color = BLACK,
        stroke_color=BLACK,
        color = BLACK
    )
    for tick in chart.get_x_axis():
        tick.set_color(BLACK)
    for tick in chart.get_y_axis():
        tick.set_color(BLACK)
    return chart

def HistogramRepresentation(scene):
    scene.add(scene.chart)
    scene.locations = [i+1 for i in range(N)]
    scene.points = [Circle(0.02, PINK).shift(RIGHT*(d-1/2)*scale + DOWN*3) for d in scene.locations]
    # scene.play(*[Create(point) for point in scene.points])
    all_points = []
    updated_points = [0]*scale*2

    for i, p in enumerate(scene.locations):
        updated_points[p*2] += scene.start_values[i]
        all_points.append(copy.copy(updated_points))
    scene.weights = updated_points
    animations = []
    for i in range(N):
        new_chart = create_barchart(scene, all_points[i])
        animations.append(Transform(scene.chart, new_chart))
        animations.append(Wait(0.3))
    t = 1.5
    scene.lag_ratio = t/N
    scene.play(
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=t),
    )

def DisplayTotalMass(scene):
    scene.h_tex = Tex("$m=$", font_size=46, color =BLACK).shift(LEFT*2+UP*1)
    scene.h_decimal = DecimalNumber(
        0,
        show_ellipsis=False,
        num_decimal_places=2,
        color = BLACK,
    )
    scene.h_dot = Circle(radius = 0, color=WHITE).shift(RIGHT*scene.m)
    scene.h_decimal.add_updater(lambda d: d.set_value(scene.h_dot.get_center()[0]))
    scene.h_decimal.next_to(scene.h_tex, RIGHT)
    scene.play(Create(scene.h_tex), Create(scene.h_decimal))

def AddMass(scene):
    m = 0.5
    # all_points = scene.weights
    scene.weights[2*2] += m
    new_chart = create_barchart(scene, scene.weights)
    animations = []
    animations.append(Transform(scene.chart, new_chart))
    time = 1.5
    scene.lag_ratio = 0.1
    scene.play(scene.h_dot.animate(run_time=time).shift(RIGHT*m),
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )
def Softmax_scene(scene):
    values = [scene.weights[(i+1)*2] for i in range(4)]
    def softmax(x, norm_sum):
        return math.exp(x) / norm_sum

    norm_sum = sum([math.exp(x_i) for x_i in values])
    x_new = []
    for x_i in values:
        x_new.append(softmax(x_i, norm_sum))
    scene.weights = [x_new[i//2-1] if i in [2,4,6,8] else scene.weights[i] for i in range(len(scene.weights))]
    m = 1.5-sum(scene.weights)

    new_chart = create_barchart(scene, scene.weights)
    animations = []
    animations.append(Transform(scene.chart, new_chart))
    time = 1.5
    scene.lag_ratio = 0.1
    scene.play(scene.h_dot.animate(run_time=time).shift(LEFT*m),
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )


def show_both(scene):
    start_values = [0.2, 0.1, 0.3, 0.4]
    values = [start_values[i//2-1] if i in [2,4,6,8] else 0 for i in range(scale*2)]
    first_chart = create_barchart(scene, values).shift(LEFT*3.3)
    time = 1
    scene.play(Create(first_chart, run_time=time), 
               scene.chart.animate(run_time=time).shift(RIGHT*3.3),
               scene.h_tex.animate(run_time=time).shift(RIGHT*3.3),
               scene.h_decimal.animate(run_time=time).shift(RIGHT*3.3),
               )