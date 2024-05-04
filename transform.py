import math

from quaternion import conjugate_quaternion, multiply_quaternions


def translation(
    point: tuple[float, float, float], translation_vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Perform translation of `point` by `translation_vector`."""
    x, y, z = point
    v1, v2, v3 = translation_vector
    return (x + v1, y + v2, z + v3)


def axial_rotation(
    point: tuple[float, float, float],
    angle_in_rads: float,
    axis_of_rotation: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`."""
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord**2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    #   Quaternion associated to point.
    p = (0, x, y, z)
    #   Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    #   Quaternion associated to image point
    q_star = conjugate_quaternion(q)
    p_prime = multiply_quaternions(q, multiply_quaternions(p, q_star))
    #   Interpret as 3D point (i.e. drop first coordinate)
    return p_prime[1], p_prime[2], p_prime[3]


def translation_then_axialrotation(
    point: tuple[float, float, float], parameters: tuple[float, ...]
):
    """Apply to `point` a translation followed by an axial rotation, both defined by `parameters`."""
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord**2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    x, y, z = translation(point=(x, y, z), translation_vector=(t1, t2, t3))
    x, y, z = axial_rotation(
        point=(x, y, z), angle_in_rads=angle_in_rads, axis_of_rotation=(v1, v2, v3)
    )
    return x, y, z
