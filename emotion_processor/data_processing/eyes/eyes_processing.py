import warnings
import numpy as np
class EyesPointsProcessing:
    def __init__(self):
        self.eyes: dict = {}
        
    def claculate_eye_arch(self, eyes_points):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)

            # Encuentra los puntos más externos de las cejas
            leftmost_point = eyes_points[0]
            rightmost_point = eyes_points[8]

            # Calcula el ángulo de inclinación
            delta_x = rightmost_point[0] - leftmost_point[0]
            delta_y = rightmost_point[1] - leftmost_point[1]
            angle = np.arctan2(delta_y, delta_x)

            # Rota los puntos para alinear la línea base con el eje horizontal
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]
            ])
            rotated_points = [np.dot(rotation_matrix, point) for point in eyes_points]

            # Extrae las coordenadas x e y de los puntos rotados
            x = [point[0] for point in rotated_points]
            y = [point[1] for point in rotated_points]

            # Ajuste polinomial
            z = np.polyfit(x, y, 2)
        return z[0] * -1000 # El coeficiente 'a' del polinomio y = ax^2 + bx + c
    
    def calculate_distances(self, eyes_points):
        left_eye_height = np.linalg.norm(np.array(eyes_points['left_eye_heigth'][0]) - np.array(eyes_points['left_eye_heigth'][1]))
        right_eye_height = np.linalg.norm(np.array(eyes_points['right_eye_heigth'][0]) - np.array(eyes_points['right_eye_heigth'][1]))
        return left_eye_height, right_eye_height
        


    def main(self, eyes_points: dict):
        self.eyes['right_eyebrow_arch'] = self.claculate_eye_arch(eyes_points['right_eye'])
        self.eyes['left_eyebrow_arch'] = self.claculate_eye_arch(eyes_points['left_eye'])
        self.eyes['left_eye_heigth'], self.eyes['right_eye_heigth'] = self.calculate_distances(eyes_points)
        print(self.eyes['left_eye_heigth'])
        return self.eyes