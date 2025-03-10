import math
import pygame

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Hexagon setup
center_x, center_y = WIDTH // 2, HEIGHT // 2
hex_radius = 200
num_vertices = 6
vertices = []
for i in range(num_vertices):
    angle = 2 * math.pi * i / num_vertices
    x = center_x + hex_radius * math.cos(angle)
    y = center_y + hex_radius * math.sin(angle)
    vertices.append((x, y))

theta = 0.0  # rotation angle
rotation_speed = 0.01  # radians per frame

# Ball setup
ball_radius = 20
ball_x = center_x + 150  # Starting position
ball_y = center_y
vx = 5.0  # Initial velocity
vy = 3.0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Update rotation angle
    theta += rotation_speed

    # Rotate vertices
    rotated_vertices = []
    for (x0, y0) in vertices:
        dx = x0 - center_x
        dy = y0 - center_y
        rot_dx = dx * math.cos(theta) - dy * math.sin(theta)
        rot_dy = dx * math.sin(theta) + dy * math.cos(theta)
        new_x = center_x + rot_dx
        new_y = center_y + rot_dy
        rotated_vertices.append((new_x, new_y))

    # Draw hexagon
    pygame.draw.polygon(screen, (255, 255, 255), rotated_vertices, 2)

    # Update ball position
    ball_x += vx
    ball_y += vy

    # Collision detection and response
    collision = False
    for i in range(num_vertices):
        p0 = rotated_vertices[i]
        p1 = rotated_vertices[(i + 1) % num_vertices]
        x0, y0 = p0
        x1, y1 = p1
        vx_edge = x1 - x0
        vy_edge = y1 - y0

        # Vector from P0 to ball's center
        cx, cy = ball_x, ball_y
        dx = cx - x0
        dy = cy - y0
        dot = dx * vx_edge + dy * vy_edge
        len_sq = vx_edge**2 + vy_edge**2

        if len_sq == 0:
            continue

        t = dot / len_sq
        if t < 0:
            closest_x, closest_y = x0, y0
        elif t > 1:
            closest_x, closest_y = x1, y1
        else:
            closest_x = x0 + t * vx_edge
            closest_y = y0 + t * vy_edge

        dx_closest = cx - closest_x
        dy_closest = cy - closest_y
        dist_sq = dx_closest**2 + dy_closest**2

        if dist_sq <= ball_radius**2:
            # Compute normal vector
            nx = vy_edge
            ny = -vx_edge
            inv_len = 1.0 / math.sqrt(len_sq)
            nx *= inv_len
            ny *= inv_len

            # Reflect velocity
            vel_dot_n = vx * nx + vy * ny
            vx_new = vx - 2 * vel_dot_n * nx
            vy_new = vy - 2 * vel_dot_n * ny

            # Move ball out of collision
            d = math.sqrt(dist_sq)
            delta = ball_radius - d
            ball_x += nx * delta
            ball_y += ny * delta

            # Update velocity
            vx, vy = vx_new, vy_new
            collision = True
            break  # Process first collision only

    # Boundary checks (optional)
    if ball_x - ball_radius < 0 or ball_x + ball_radius > WIDTH:
        vx = -vx
    if ball_y - ball_radius < 0 or ball_y + ball_radius > HEIGHT:
        vy = -vy

    # Draw ball
    pygame.draw.circle(screen, (255, 0, 0), (int(ball_x), int(ball_y)), ball_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
