import pygame
import numpy as np
from config import *


class Renderer:
    """Класс для отрисовки игровых объектов"""

    @staticmethod
    def create_transparent_surface(width, height, color):
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(color)
        return surface

    @staticmethod
    def draw_agent(surface, agent):
        if agent.caught:
            Renderer._draw_caught_agent(surface, agent)
        elif agent.escaped:
            Renderer._draw_escaped_agent(surface, agent)
        else:
            Renderer._draw_normal_agent(surface, agent)

        if agent.is_hunter and agent.capture_target:
            Renderer._draw_capture_effects(surface, agent)

    @staticmethod
    def _draw_caught_agent(surface, agent):
        pygame.draw.circle(surface, (100, 100, 100), (int(agent.x), int(agent.y)), agent.radius)
        pygame.draw.line(surface, (200, 0, 0),
                         (agent.x - agent.radius // 2, agent.y - agent.radius // 2),
                         (agent.x + agent.radius // 2, agent.y + agent.radius // 2), 3)
        pygame.draw.line(surface, (200, 0, 0),
                         (agent.x + agent.radius // 2, agent.y - agent.radius // 2),
                         (agent.x - agent.radius // 2, agent.y + agent.radius // 2), 3)

    @staticmethod
    def _draw_escaped_agent(surface, agent):
        pygame.draw.circle(surface, (0, 200, 0), (int(agent.x), int(agent.y)), agent.radius)
        pygame.draw.line(surface, (255, 255, 255),
                         (agent.x - agent.radius // 2, agent.y),
                         (agent.x, agent.y + agent.radius // 2), 3)
        pygame.draw.line(surface, (255, 255, 255),
                         (agent.x, agent.y + agent.radius // 2),
                         (agent.x + agent.radius // 2, agent.y - agent.radius // 3), 3)

    @staticmethod
    def _draw_normal_agent(surface, agent):
        color = (255, 255, 0) if agent.fixing_generator else agent.color
        pygame.draw.circle(surface, color, (int(agent.x), int(agent.y)), agent.radius)

        # Глаза
        eye_offset = 4
        eye_radius = 3
        look_dx = np.cos(np.radians(agent.vision_direction))
        look_dy = np.sin(np.radians(agent.vision_direction))

        left_eye_x = agent.x - eye_offset + look_dx * 3
        left_eye_y = agent.y - eye_offset + look_dy * 3
        right_eye_x = agent.x + eye_offset + look_dx * 3
        right_eye_y = agent.y - eye_offset + look_dy * 3

        pygame.draw.circle(surface, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)

        # Зрачки
        pupil_offset = 1.5
        left_pupil_x = left_eye_x + look_dx * pupil_offset
        left_pupil_y = left_eye_y + look_dy * pupil_offset
        right_pupil_x = right_eye_x + look_dx * pupil_offset
        right_pupil_y = right_eye_y + look_dy * pupil_offset

        pygame.draw.circle(surface, (0, 0, 0), (int(left_pupil_x), int(left_pupil_y)), eye_radius - 1)
        pygame.draw.circle(surface, (0, 0, 0), (int(right_pupil_x), int(right_pupil_y)), eye_radius - 1)

        # Эффекты
        if agent.fixing_generator and pygame.time.get_ticks() % 500 < 250:
            pygame.draw.circle(surface, (255, 255, 0), (int(agent.x), int(agent.y + 15)), 5)

        if agent.cooldown > 0:
            pygame.draw.circle(surface, (0, 100, 255), (int(agent.x), int(agent.y + 20)), 3)

    @staticmethod
    def _draw_capture_effects(surface, agent):
        distance = np.sqrt((agent.capture_target.x - agent.x) ** 2 + (agent.capture_target.y - agent.y) ** 2)
        if distance < CAPTURE_RADIUS:
            pulse = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
            line_width = int(2 + pulse * 2)
            line_color = (255, int(100 + pulse * 100), 50)

            pygame.draw.line(surface, line_color,
                             (int(agent.x), int(agent.y)),
                             (int(agent.capture_target.x), int(agent.capture_target.y)), line_width)

            progress = agent.hold_steps / CAPTURE_HOLD_STEPS
            if progress > 0:
                progress_color = (255, int(100 + pulse * 100), 50, 150)
                progress_surface = pygame.Surface((agent.capture_target.radius * 2 + 10,
                                                   agent.capture_target.radius * 2 + 10),
                                                  pygame.SRCALPHA)

                progress_angle = int(360 * progress)
                pygame.draw.arc(progress_surface, progress_color,
                                (5, 5, agent.capture_target.radius * 2, agent.capture_target.radius * 2),
                                0, np.radians(progress_angle), 4)

                surface.blit(progress_surface,
                             (agent.capture_target.x - agent.capture_target.radius - 5,
                              agent.capture_target.y - agent.capture_target.radius - 5))

    @staticmethod
    def draw_vision_cone(surface, agent):
        if agent.caught or agent.escaped:
            return

        cone_surface = pygame.Surface((agent.vision_radius * 2, agent.vision_radius * 2), pygame.SRCALPHA)

        start_angle = agent.vision_direction - agent.vision_angle / 2
        end_angle = agent.vision_direction + agent.vision_angle / 2

        points = []
        num_segments = 60
        center = (agent.vision_radius, agent.vision_radius)

        for i in range(num_segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_segments
            rad = np.radians(angle)
            x = agent.vision_radius + agent.vision_radius * np.cos(rad)
            y = agent.vision_radius + agent.vision_radius * np.sin(rad)
            points.append((x, y))

        for i in range(len(points) - 1):
            segment_points = [center, points[i], points[i + 1]]

            dist_factor = min(1.0, np.sqrt(
                (points[i][0] - center[0]) ** 2 + (points[i][1] - center[1]) ** 2) / agent.vision_radius)
            alpha = int(80 * (1 - dist_factor * 0.7))

            if agent.is_hunter:
                color = (255, 100, 100, alpha)
            else:
                color = (100, 100, 255, alpha)

            if len(segment_points) >= 3:
                pygame.draw.polygon(cone_surface, color, segment_points)

        outline_points = points
        if len(outline_points) > 1:
            if agent.is_hunter:
                outline_color = (255, 50, 50, 150)
            else:
                outline_color = (50, 50, 255, 150)
            pygame.draw.lines(cone_surface, outline_color, False, outline_points, 2)

        surface.blit(cone_surface, (agent.x - agent.vision_radius, agent.y - agent.vision_radius))

        end_x = agent.x + agent.vision_radius * 0.8 * np.cos(np.radians(agent.vision_direction))
        end_y = agent.y + agent.vision_radius * 0.8 * np.sin(np.radians(agent.vision_direction))

        if agent.is_hunter:
            direction_color = (255, 100, 100, 200)
        else:
            direction_color = (100, 100, 255, 200)

        pygame.draw.line(surface, direction_color, (agent.x, agent.y), (end_x, end_y), 3)
        pygame.draw.circle(surface, direction_color, (int(end_x), int(end_y)), 4)