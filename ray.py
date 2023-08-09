import numpy as np
from PIL import Image
import random
from utils import *
"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


def load_sand():
    image = Image.open('sand.png')
    return np.asfarray(image) / 255


def load_water():
    image = Image.open('water.png')
    return np.asfarray(image) / 255


def load_pyramid():
    image = Image.open('pyramid.png').convert('RGB')
    return np.asfarray(image) / 255


def load_sun():
    image = Image.open('sun.png').convert('RGB')
    return np.asfarray(image) / 255


SAND = load_sand()
WATER = load_water()
PYRAMID = load_pyramid()
SUN = load_sun()
IDENTITY_TEXTURE = np.zeros((5, 5))


class Ray:

    def __init__(self, origin, direction, start=0.0, end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0.0, p=20.0, k_m=0.0, k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material, texture=IDENTITY_TEXTURE):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material
        self.texture = texture

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        start_point = ray.origin + ray.start * ray.direction
        d = ray.direction
        ec = start_point - self.center
        det = np.square(np.dot(
            d, ec)) - (np.dot(d, d) *
                       (np.dot(ec, ec) - np.square(self.radius)))
        t = 0
        if det < 0:
            return no_hit
        elif det == 0:
            t = np.dot(-d, ec) / np.dot(d, d) + ray.start
        elif np.linalg.norm(start_point - self.center) < self.radius:
            s2 = (np.dot(-d, ec) + np.sqrt(det)) / np.dot(d, d)
            t = s2 + ray.start
        else:
            s1 = (np.dot(-d, ec) - np.sqrt(det)) / np.dot(d, d)
            t = s1 + ray.start
        if t > ray.end or t < ray.start:
            return no_hit
        point = ray.origin + d * t
        normal = (point - self.center) / self.radius
        return Hit(t, point, normal, self.material)


class Triangle:

    def __init__(self, vs, material, texture=IDENTITY_TEXTURE):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material
        self.texture = texture

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        start_point = ray.origin + ray.start * ray.direction
        d = ray.direction
        A = np.array([self.vs[0] - self.vs[1], self.vs[0] - self.vs[2], d])
        T_mtx = np.array([
            self.vs[0] - self.vs[1], self.vs[0] - self.vs[2],
            self.vs[0] - start_point
        ])
        B_mtx = np.array(
            [self.vs[0] - start_point, self.vs[0] - self.vs[2], d])
        G_mtx = np.array(
            [self.vs[0] - self.vs[1], self.vs[0] - start_point, d])
        t = (np.linalg.det(T_mtx) / np.linalg.det(A)) + ray.start
        beta = np.linalg.det(B_mtx) / np.linalg.det(A)
        gamma = np.linalg.det(G_mtx) / np.linalg.det(A)
        cond1 = t > ray.end or t < ray.start
        cond2 = gamma > 1 or gamma < 0
        cond3 = beta > 1 - gamma or beta < 0
        if cond1 or cond2 or cond3:
            return no_hit
        point = ray.origin + d * t
        # normal = np.random.randn(3)
        AB = self.vs[0] - self.vs[1]
        AC = self.vs[0] - self.vs[2]
        normal = np.cross(AB, AC)
        return Hit(t, point, normal, self.material)


class Camera:

    def __init__(
            self,
            eye=vec([0, 0, 0]),
            target=vec([0, 0, -1]),
            up=vec([0, 1, 0]),
            vfov=90.0,
            aspect=1.0,
    ):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.distance = target - eye
        self.target = target
        self.vfov = vfov
        self.height = 2 * np.linalg.norm(self.distance) * np.tan(
            np.radians(vfov / 2))
        self.width = self.height * aspect
        self.up = up

        # TODO A4 implement this constructor to store whatever you need for ray generation

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        self.up -= np.dot(self.up, normalize(self.distance)) * normalize(
            self.distance)
        horz = np.cross(self.distance, self.up)
        wc = [
            self.width * (img_point[0] - 0.5),
            self.height * (img_point[1] - 0.5)
        ]
        point = self.target + (normalize(self.up) * wc[1]) + (normalize(horz) *
                                                              wc[0])
        return Ray(self.eye, point - self.eye)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, surface):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        if hit == no_hit:
            return scene.bg_color
        r = np.linalg.norm(hit.point - self.position)
        p = hit.material.p
        l = normalize(self.position - hit.point)
        v = normalize(ray.origin - hit.point)
        h = normalize((v + l) / np.linalg.norm(v + l))
        n = normalize(hit.normal)
        return ((texture_fn(hit.point, surface, hit.material.k_d) +
                 hit.material.k_s * np.power(max(0, np.dot(n, h)), p)) *
                max(0, np.dot(n, l)) / np.power(r, 2) * self.intensity)
        # return ((hit.material.k_d +
        #          hit.material.k_s * np.power(max(0, np.dot(n, h)), p)) *
        #         max(0, np.dot(n, l)) / np.power(r, 2) * self.intensity)


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene, surface):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        if hit == no_hit:
            return scene.bg_color
        return self.intensity * hit.material.k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        min_hit = no_hit
        min_surf = None
        for s in self.surfs:
            temp_hit = s.intersect(ray)
            if temp_hit.t < min_hit.t:
                min_hit = temp_hit
                min_surf = s
        return (min_hit, min_surf)


MAX_DEPTH = 4
EPSILON = 1e-6


def texture_fn(point, surface, color):
    texture = surface.texture
    if texture == IDENTITY_TEXTURE:
        return color
    (H, W, C) = texture.shape
    if isinstance(surface, Sphere):
        normal = point - surface.center
        x = (np.pi + np.arctan2(normal[1], normal[0])) / (2 * np.pi)
        y = (np.pi - np.arccos((normal[2] / normal[0] % 1))) / np.pi
        Y, X = int(H * y), int(W * x)
        return (texture[Y, X, :])
    elif isinstance(surface, Triangle):
        verts = surface.vs
        xa = abs(verts[0][0] - verts[1][0])
        xb = abs(verts[0][0] - verts[2][0])
        ya = abs(verts[0][1] - verts[1][1])
        yb = abs(verts[0][1] - verts[2][1])
        max_diff_x = xa if xa > xb else xb
        max_diff_y = ya if ya > yb else yb
        x = abs(point[0] - verts[0][0])
        y = abs(point[1] - verts[0][1])
        if max_diff_x != 0:
            x /= max_diff_x
        if max_diff_y != 0:
            y /= max_diff_y
        Y, X = int(H * y), int(W * x)
        return (texture[Y, X, :])


def shade(ray, hit, scene, lights, surface, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function

    res = vec([0, 0, 0])
    sray = Ray(hit.point, -1 * ray.direction, EPSILON)
    if hit == no_hit:
        return scene.bg_color
    else:
        for l in lights:
            if isinstance(l, PointLight):
                sray = Ray(hit.point, l.position - hit.point, EPSILON)
                (h, _) = scene.intersect(sray)
                if h == no_hit:
                    res += l.illuminate(ray, hit, scene, surface)
            else:
                res += l.illuminate(ray, hit, scene, surface)
    d = normalize(ray.direction)
    n = normalize(hit.normal)
    r = d - 2 * np.dot(n, d) * n
    ref_ray = Ray(hit.point, r, EPSILON)
    (ref_hit, surface) = scene.intersect(ref_ray)
    if depth <= MAX_DEPTH and hit.material != None:
        res = res + hit.material.k_m * shade(ref_ray, ref_hit, scene, lights,
                                             surface, depth + 1)
    return res


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function
    res = np.zeros((ny, nx, 3), np.float32)
    for x in range(nx):
        for y in range(ny):
            ray = camera.generate_ray([x / nx, y / ny])
            (hit, surface) = scene.intersect(ray)
            res[y, x] = shade(ray, hit, scene, lights, surface)
    return res
