import numpy as np
from manim import *
import math
import torch
import copy
from manim.utils.color import Colors

scale= 9
N = 7

class OFDM_alg(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        start_values = [2.5, 2, 1.5, 0.5, 2.2, 0.7, 1.7]
        start_values = [i/8 for i in start_values]
        self.y_value = 1/sum(start_values)
        self.start_values = [i*self.y_value for i in start_values]
        self.m = sum(self.start_values)
        self.chart = create_barchart(self, [0]*scale*2)
        HistogramRepresentation(self)
        # self.wait(2)
        DisplayTotalMass(self)
        # self.wait(1)
        DrawDerivate(self)
        # self.wait(2)
        ShowSmallest(self)
        # self.wait(1)
        AddMass(self)
        # self.wait(1)
        ShowLargest(self)
        # self.wait(1)
        TakeLargestMass(self)
        # self.wait(1)
        ShowAlmostLargest(self)
        # self.wait(1)
        TakeAlmostLargestMass(self, time = 0.8)
        # self.wait(1)
    
def create_barchart(scene, values):
    chart = BarChart(
        values=values,
        bar_names=[""]*scale*2,
        y_range=[0, scene.y_value, 1/10],
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
    t = 3
    scene.lag_ratio = t/N
    scene.play(
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=t),
    )

def DisplayTotalMass(scene):
    scene.h_tex = Tex("$m=$", font_size=46, color =BLACK).shift(LEFT*2.8+UP*1)
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
def DrawDerivate(scene):
    func = derivative_func(scene, scene.chart)
    scene.play(Create(func))
    # scene.play(Transform(scene.chart, new_chart), run_time=1)

def derivative_func(scene, chart):
    return chart.plot(
        lambda t: ((t-6.5)**2/20+4)/8*scene.y_value,
        x_range=[1,scale*2-2],
        color=BLACK,
    )

def ShowSmallest(scene):
    point = Circle(0.1,BLUE, fill_opacity=1).shift(LEFT*1.25)
    scene.play(Create(point))
    scene.play(Flash(point))

def ShowLargest(scene):
    point = Circle(0.1,RED, fill_opacity=1).shift(LEFT*1.25+RIGHT*4 + UP*7**2/20+DOWN*0.05)
    scene.play(Create(point))
    scene.play(Flash(point))

def ShowAlmostLargest(scene):
    point = Circle(0.1,RED, fill_opacity=1).shift(LEFT*1.25+RIGHT*3 + UP*(5)**2/20 + UP*0.1)
    scene.play(Create(point))
    scene.play(Flash(point))

def AddMass(scene):
    m = 2
    # all_points = scene.weights
    scene.weights[3*2] += m*scene.y_value/8
    # new_chart = BarChart(
    #     values=all_points,
    #     bar_names=[""]*scale*2,
    #     y_range=[0, 8, 1],
    #     x_length=scale,
    #     y_length=6,
    #     bar_width=0.5, 
    #     bar_fill_opacity=0.5,
    #     bar_colors = [BLACK],
    #     fill_color = BLACK,
    #     stroke_color=BLACK,
    # )
    w_1 = 0.05
    w_2 = m*3/4
    rect_1 = Rectangle(width = 0.5/2.2, height = w_1, color = BLACK, fill_color=BLUE, fill_opacity=0.8).shift(LEFT*1.25 + DOWN*3 + UP*(1.5*3/4 + w_1/2))
    rect_2 = Rectangle(width = 0.5/2.2, height = w_2, color = BLACK, fill_color=BLUE, fill_opacity=0.8).shift(LEFT*1.25 + DOWN*3 + UP*(1.5*3/4 + w_2/2))
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
    # scene.play(Transform(scene.chart, new_chart))
    
def TakeLargestMass(scene):
    m = 1.7
    scene.weights[7*2] -= m*scene.y_value/8
    w_1 = 0.05
    w_2 = m*3/4
    rect_1 = Rectangle(width = 0.5/2.2, height = w_1, color = BLACK, fill_color=RED, fill_opacity=0.8).shift(RIGHT*2.75 + DOWN*3 + UP*(m*3/4 - w_1/2))
    rect_2 = Rectangle(width = 0.5/2.2, height = w_2, color = BLACK, fill_color=RED, fill_opacity=0.8).shift(RIGHT*2.75 + DOWN*3 + UP*(m*3/4 - w_2/2))
    scene.play(Create(rect_1))
    animations = []
    animations.append(Transform(rect_1, rect_2))
    time = 1.5
    scene.lag_ratio = 0.1
    scene.play(scene.h_dot.animate(run_time=time).shift(LEFT*m*scene.y_value/8),
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )

def TakeAlmostLargestMass(scene, time=1.5):
    m = 0.3
    scene.weights[6*2] -= m*scene.y_value/8
    w_1 = 0.05
    w_2 = m*3/4
    rect_1 = Rectangle(width = 0.5/2.2, height = w_1, color = BLACK, fill_color=RED, fill_opacity=0.8).shift(RIGHT*1.75 + DOWN*3 + UP*(0.7*3/4 - w_1/2))
    rect_2 = Rectangle(width = 0.5/2.2, height = w_2, color = BLACK, fill_color=RED, fill_opacity=0.8).shift(RIGHT*1.75 + DOWN*3 + UP*(0.7*3/4 - w_2/2))
    scene.play(Create(rect_1))
    animations = []
    animations.append(Transform(rect_1, rect_2))
    scene.lag_ratio = 0.1
    scene.play(scene.h_dot.animate(run_time=time).shift(LEFT*m*scene.y_value/8),
        Succession(
        *animations,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )

def NormalKernelsAndKDEFunction(scene):
    scene.kernels = [create_kernel(scene.chart, d, scene.h) for d in data]
    animations_1 = [Create(kernel) for kernel in scene.kernels]

    scene.play(animations_1[0], run_time=2)
    scene.wait(1)

    segmented_func = [create_segmented_func(scene.chart, k+1, scene.h) for k in range(N-1)]
    animations_2 = []
    scene.play(Create(segmented_func[0]))
    for func in segmented_func[1:]: 
        animations_2.append(Transform(segmented_func[0], func))
        animations_2.append(Wait(0.3))

    scene.play(LaggedStart(
        *animations_1[1:],
        lag_ratio=scene.lag_ratio,
        run_time=3),

        Succession(
        *animations_2,
        lag_ratio=scene.lag_ratio,
        run_time=3),
    )
    scene.KDEFunc = segmented_func[0]


def HideBars(scene):
    new_chart = BarChart(
        values=[0]*11,
        bar_names=[""]*11,
        y_range=[0, 8, 1],
        x_length=11,
        y_length=6,
        bar_width=1, 
        bar_fill_opacity=0,
    )
    scene.play(Transform(scene.chart, new_chart), run_time=1)
    
def CreateTexhAndCircle(scene):
    scene.h_tex = Tex("$h=$", font_size=46).shift(LEFT*2.8+UP*1)
    scene.h_decimal = DecimalNumber(
        0,
        show_ellipsis=True,
        num_decimal_places=3,
    )
    scene.h_dot = Circle(radius = 0, color=BLACK).shift(RIGHT*scene.h)
    scene.h_decimal.add_updater(lambda d: d.set_value(scene.h_dot.get_center()[0]))
    scene.h_decimal.next_to(scene.h_tex, RIGHT)
    scene.play(Create(scene.h_tex), Create(scene.h_decimal))

def ScaleKDEDownAndUp(scene):
    animations_1 = []
    animations_2 = []
    n = 30
    for i in range(n):
        increment = 0.015
        scene.h += increment
        func = create_KDE(scene.chart, scene.h)
        animations_1.append(scene.h_dot.animate.shift(RIGHT*increment))
        animations_1.append(Wait(0.1))
        animations_2.append(Transform(scene.KDEFunc, func))
        animations_2.append(Wait(0.1))

    time = 4
    scene.lag_ratio = time/(n)
    scene.play(scene.h_dot.animate(run_time=time).shift(RIGHT*n*increment),
        Succession(
        *animations_2,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )

    scene.wait(1)
    animations_2 = []
    n=55
    for i in range(n):
        increment = -0.015
        scene.h += increment
        func = create_KDE(scene.chart, scene.h)
        animations_1.append(scene.h_dot.animate.shift(RIGHT*increment))
        animations_1.append(Wait(0.1))
        animations_2.append(Transform(scene.KDEFunc, func))
        animations_2.append(Wait(0.1))
    time = 5
    scene.lag_ratio = time/(n)
    scene.play(scene.h_dot.animate(run_time=time).shift(RIGHT*n*increment),
        Succession(
        *animations_2,
        lag_ratio=scene.lag_ratio,
        run_time=time),
    )