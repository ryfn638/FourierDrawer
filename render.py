## the plan
import numpy as np
from PIL import Image
from scipy.fft import fftfreq
import pygame
# Fourier Series Stuff
import time
import algorithm
import cv2

image_path = "output_0251.jpg"
image_input = cv2.imread(image_path)
image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)

height, width = image_input.shape


num_frames = 6572

pygame.init()
screen = pygame.display.set_mode((width, height))


def getfilepath(number):
    # Things are padded up until number 1000 therefore
    return str(number).zfill(4)

# cheers to u/plastic_astronomer for this draw arrow function
def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)


# THiS IS CORRECT, i think???
def arrow_positions_with_reconstruction(coeffs, t=0):
    
    positions = [(0 + 0j)]

    magnitude = np.abs(coeffs[0])
    dir = np.angle(coeffs[0])

    positions.append(positions[-1] + magnitude*np.exp(1j*dir))

    # Add the DC Coefficient
    N = len(coeffs)
    freqs = fftfreq(N)
    for k in range(N // 2):
        w = 2 * np.pi * freqs[k+1]
        p_coeff = coeffs[k+1]
        p_mag = np.abs(p_coeff)
        p_dir = np.angle(p_coeff)

        positions.append(positions[-1] + p_mag*np.exp(1j*(p_dir + w * t)))

        w = 2 * np.pi * freqs[-(k+1)]
        n_coeff = coeffs[-(k+1)]
        n_mag = np.abs(n_coeff)
        n_dir = np.angle(n_coeff)


        positions.append(positions[-1] + n_mag*np.exp(1j*(n_dir + w * t)))

    return positions



def main(image):
    running = True
    covered_pixels = []

    max_coeffs = 300

    # THis is right
    coeffs = algorithm.findcoeffs(image, max_coeffs)

    t_elapsed = 0
    dt = 1 * np.pi/180
    
    while running:
        screen.fill((0, 0, 0))
        # This WOrks Fine
        for pixel in covered_pixels:
            screen.set_at((int(pixel.x), int(pixel.y)), (255,0,0))

        # This works
        arrow_positions= arrow_positions_with_reconstruction(coeffs, t=t_elapsed*dt)


        for index in range(len(arrow_positions)-1):  # -1 because we access index+1
            start_pos = pygame.Vector2(0, height) - pygame.Vector2(np.real(arrow_positions[index]),
                                    np.imag(arrow_positions[index])) - pygame.Vector2(-width//2, height//2)
            
            end_pos = pygame.Vector2(0, height) - pygame.Vector2(np.real(arrow_positions[index+1]),
                                    np.imag(arrow_positions[index+1])) - pygame.Vector2(-width//2, height//2)
            
            draw_arrow(
                surface=screen,
                start=start_pos,
                end=end_pos,
                color=pygame.Color(255, 255, 255),
            )


        position_end = pygame.Vector2(0, height) - pygame.Vector2(np.real(arrow_positions[-1]),
                                    np.imag(arrow_positions[-1])) - pygame.Vector2(-width//2, height//2)
        

        if (position_end not in covered_pixels):
            covered_pixels.append(position_end)
        
        t_elapsed += 3
        #time.sleep(0.000001)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False







main(image_input)

pygame.quit()





