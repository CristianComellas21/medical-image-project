def multiply_quaternions(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Multiply two quaternions, expressed as (1, i, j, k)."""
    # Your code here:
    #   ...
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    )


def conjugate_quaternion(
    q: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Multiply two quaternions, expressed as (1, i, j, k)."""
    # Your code here:
    #   ...
    return (q[0], -q[1], -q[2], -q[3])
