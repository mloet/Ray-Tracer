from utils import *
from ray import *
from cli import render

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
tan2 = Material(vec([0.65, 0.65, 0.35]), 0.4)
blue = Material(vec([0.2, 0.3, 1]), 0.5)

sun = Material(vec([1, 0.6, 0.0]), 250)

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
vs_list = 0.5 * read_obj_triangles(open("pyramid.obj")) + [1, 0, 0]
vs_list2 = 0.4 * read_obj_triangles(open("pyramid.obj")) + [0.5, -0.1, 1.2]
vs_list3 = 0.3 * read_obj_triangles(open("pyramid.obj")) + [1, 0, -1.5]

scene = Scene([
    # Make a big sphere for the floor
    Sphere(vec([3, -39.7, 0]), 39.5, tan, SAND),
] + [
    # Make a big sphere for the floor
    Sphere(vec([3.1, -39.95, -6]), 39.5, tan, SAND),
] + [
    # Make a big sphere for the floor
    Sphere(vec([-6.5, -40.6, -6]), 39.5, tan, SAND),
] + [
    # Make a big sphere for the floor
    Sphere(vec([-0.5, -39.85, 0]), 39.5, blue, WATER),
] + [
    # Make a big sphere for the floor
    Sphere(vec([2, -50.1, -5.2]), 49.5, blue, WATER),
] + [
    # Sphere for sun
    Sphere(vec([-16, -1.05, -12]), 1, sun, SUN),
] + [
    # Make a big sphere for the floor
    Sphere(vec([-3, -39.9, 0]), 39.5, tan, SAND),
] + [
    # Make triangle objects from the vertex coordinates
    Triangle(vs, tan2, PYRAMID) for vs in vs_list
] + [
    # Make triangle objects from the vertex coordinates
    Triangle(vs, tan2, PYRAMID) for vs in vs_list2
] + [
    # Make triangle objects from the vertex coordinates
    Triangle(vs, tan2, PYRAMID) for vs in vs_list3
])

lights = [
    PointLight(vec([-7.5, 0, -5.5]), vec([200, 150, 150])),
    PointLight(vec([0, 10, 0]), vec([15, 15, 15])),
    AmbientLight(0.2),
]

# camera = Camera(vec([0.1, 40, 0.1]), target=vec(
#     [0, 0, 0]), vfov=25, aspect=16 / 9)
camera = Camera(vec([3, 1.7, 5]),
                target=vec([0, 0, 0]),
                vfov=25,
                aspect=16 / 9)

render(camera, scene, lights)
