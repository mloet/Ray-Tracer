from utils import *
from ray import *
from cli import render

tan = Material(vec([0.4, 0.4, 0.2]), k_s=0.3, p=90, k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.4)

camera = Camera(vec([3, 1.2, 5]),
                target=vec([0, -0.4, 0]),
                vfov=24,
                aspect=16 / 9)
ray = camera.generate_ray(vec([0.5, 0.5]))

scene = Scene([
    Sphere(ray.origin + 2 * ray.direction, 0.3, tan),
    # Sphere(vec([0, .2, 0]), 0.2, tan),
    # Sphere(vec([0, -.3, 0]), 0.6, tan),
    # Sphere(vec([0, -42, 0]), 39.5, gray),
])

lights = [
    PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

render(camera, scene, lights)
