import numpy as np
import warnings

class EyebrowsPointsProcessing:
    def __init__(self):
        self.eyebrows: dict = {}


    def claculate_eyebrow_arch(self, eyebrow_points):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)

            # Encuentra los puntos más externos de las cejas
            leftmost_point = eyebrow_points[0]
            rightmost_point = eyebrow_points[4]

            # Calcula el ángulo de inclinación
            delta_x = rightmost_point[0] - leftmost_point[0]
            delta_y = rightmost_point[1] - leftmost_point[1]
            angle = np.arctan2(delta_y, delta_x)

            # Rota los puntos para alinear la línea base con el eje horizontal
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]
            ])
            rotated_points = [np.dot(rotation_matrix, point) for point in eyebrow_points]

            # Extrae las coordenadas x e y de los puntos rotados
            x = [point[0] for point in rotated_points]
            y = [point[1] for point in rotated_points]

            # Ajuste polinomial
            z = np.polyfit(x, y, 2)
        return z[0] * -1000 # El coeficiente 'a' del polinomio y = ax^2 + bx + c
    

    def calculate_distances(self, eyebrows_points):
        distance_between_eyebrows = np.linalg.norm(np.array(eyebrows_points['distance_between_eyebrows'][0]) - np.array(eyebrows_points['distance_between_eyebrows'][1]))
        distance_right_eyebrow_eye = np.linalg.norm(np.array(eyebrows_points['distance_right_eyebrow_eye'][0]) - np.array(eyebrows_points['distance_right_eyebrow_eye'][1]))
        distance_left_eyebrow_eye = np.linalg.norm(np.array(eyebrows_points['distance_left_eyebrow_eye'][0]) - np.array(eyebrows_points['distance_left_eyebrow_eye'][1]))
        distance_right_forehead = np.linalg.norm(np.array(eyebrows_points['distance_right_forehead'][0]) - np.array(eyebrows_points['distance_right_forehead'][1]))
        distance_left_forehead = np.linalg.norm(np.array(eyebrows_points['distance_left_forehead'][0]) - np.array(eyebrows_points['distance_left_forehead'][1]))

        return distance_between_eyebrows, distance_right_eyebrow_eye, distance_left_eyebrow_eye, distance_right_forehead, distance_left_forehead


    def main(self, eyebrows_points: dict):
        self.eyebrows['right_eyebrow_arch'] = self.claculate_eyebrow_arch(eyebrows_points['right_eyebrow'])
        self.eyebrows['left_eyebrow_arch'] = self.claculate_eyebrow_arch(eyebrows_points['left_eyebrow'])
        self.eyebrows['distance_between_eyebrows'], self.eyebrows['distance_right_eyebrow_eye'], self.eyebrows['distance_left_eyebrow_eye'], self.eyebrows['distance_right_forehead'], self.eyebrows['distance_left_forehead'] = self.calculate_distances(eyebrows_points)
        return self.eyebrows
