import numpy as np
from manim import *
import math
import torch
import copy

N = 20
scale = 11

def get_data():
    torch.manual_seed(15)
    data, _ = torch.rand(N).sort()
    data = data.tolist()
    return data

def K(x, h):
    return 1/(np.sqrt(2*np.pi))*np.exp(-x**2/2)

def KDE(x, h):
    return 1/(N*h) *sum([K((x-d*scale)/h, h) for d in data])

def KDE_custom(x, k, h):
    return 1/(N*h) *sum([K((x-d*scale)/h, h) for d in data[:k]])

def create_KDE(chart, h):
    return chart.plot(
        lambda t: KDE(t, h)*7,
        x_range=[0,11],
        color=RED,
    )

def create_kernel(chart, d, h):
    return chart.plot(
        lambda t: K((t-scale*d)*7, h),
        x_range=[-0.6+scale*d,0.6+scale*d],
        color=BLUE,
    )

def create_segmented_func(chart, k, h):
    return chart.plot(
        lambda t: KDE_custom(t, k, h)*7,
        x_range=[0,11],
        color=RED,
    )

data = get_data()

class KDEScene(Scene):
    def construct(self):
        self.chart = BarChart(
            values=[0]*11,
            bar_names=[""]*11,
            y_range=[0, 8, 1],
            x_length=11,
            y_length=6,
            bar_width=1, 
            bar_fill_opacity=0,
        )
        self.h = math.ceil(10*1.06*N**(-1/5))/10
        DotsAndHistogramRepresentation(self)
        self.wait(2)
        NormalKernelsAndKDEFunction(self)
        self.wait(2)
        HideBars(self)
        self.wait(1)
        CreateTexhAndCircle(self)
        self.wait(1)
        ScaleKDEDownAndUp(self)
        self.wait(3)
    


def DotsAndHistogramRepresentation(scene):
    scene.points = [Circle(0.02, PINK).shift(RIGHT*(d-1/2)*scale + DOWN*3) for d in data]
    scene.add(scene.chart)
    scene.play(*[Create(point) for point in scene.points])
    all_points = []
    updated_points = [0]*11
    for p in data:
        updated_points[math.floor(p*scale)] += 1
        all_points.append(copy.copy(updated_points))
    animations_1 = []
    animations_2 = []
    for i, p in enumerate(scene.points):
        new_chart = BarChart(
            values=all_points[i],
            bar_names=[""]*11,
            y_range=[0, 8, 1],
            x_length=11,
            y_length=6,
            bar_width=1, 
            bar_fill_opacity=0,
        )
        animations_1.append(Flash(p, run_time=0.3))
        animations_2.append(Transform(scene.chart, new_chart))
        animations_2.append(Wait(0.3))
    scene.lag_ratio = 5/N
    scene.play(LaggedStart(
        *animations_1,
        lag_ratio=scene.lag_ratio,
        run_time=5),

        Succession(
        *animations_2,
        lag_ratio=scene.lag_ratio,
        run_time=5),
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