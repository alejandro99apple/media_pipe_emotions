import numpy as np
import cv2
import mediapipe as mp
from typing import Any, Tuple

class FaceMeshMediaPipe:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils  #Objeto de dibujo de mediapipe
        self.config_draw = self.mp_draw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)  #Configuración de dibujo

        self.mp_face_mesh_object = mp.solutions.face_mesh  #Objeto de detección de puntos de la cara
        self.face_mesh_mp = self.mp_face_mesh_object.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.6,min_tracking_confidence=0.6, refine_landmarks=True)  #Configuración de detección de puntos de la cara

        self.eye_points: dict = {}
        self.eyebrow_points: dict = {}
        self.mouth_points: dict = {}
        self.nose_points: dict = {}

        self.mesh_points: list = []


    def face_mesh_inference(self, face_image:np.ndarray) -> Tuple[bool, Any]:       #Dice si la imagen se puede procesar y devuelve los puntos de la cara
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  #Conversión de la imagen a RGB

        face_mesh = self.face_mesh_mp.process(rgb_image)
        if face_mesh.multi_face_landmarks is None:
            return  False, face_mesh
        else:
            return  True, face_mesh
        



    def extract_face_mesh_points(self, face_image:np.ndarray, face_mesh_info:Any):  #Extrae los puntos de los ojos, cejas, boca y nariz
        h, w, c = face_image.shape  #Dimensiones de la imagen
        self.mesh_points = []
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for i, point in enumerate(face_mesh.landmark):
                x, y = int(point.x*w), int(point.y*h)   
                self.mesh_points.append([i, x, y])  #Guarda los puntos de la cara

        return self.mesh_points


    def draw_face_mesh(self, face_image:np.ndarray, face_mesh_info:Any, color: Tuple[int, int, int]):  #Visualiza los puntos
        self.config_draw = self.mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=1)  #Configuración de dibujo
        for face_mesh in face_mesh_info.multi_face_landmarks:
            self.mp_draw.draw_landmarks(face_image, face_mesh, self.mp_face_mesh_object.FACEMESH_TESSELATION, self.config_draw, self.config_draw)  #Dibuja los puntos de la cara



    def extract_eye_points(self, face_points: list, face_image: np.ndarray) -> dict:  #Extrae los puntos de los ojos
        self.eye_points = {'right_eye': [], 'left_eye': [], 'left_eye_distances': [], 'right_eye_distances': []}
        if len(face_points) == 478:

            right_eye_index = [33, 246, 161, 160, 159, 158, 157, 173, 133]
            left_eye_index = [263, 466, 388, 387, 386, 385, 384, 398, 362]
            lef_eye_distances_index = [159, 145, 27, 230]
            right_eye_distances_index = [386, 374, 257, 450]

            def get_eye_points(index):
                return [face_points[i][1:] for i in index]

            right_eye_points = get_eye_points(right_eye_index)
            left_eye_points = get_eye_points(left_eye_index)
            left_eye_distances_points = get_eye_points(lef_eye_distances_index)
            right_eye_distances_points = get_eye_points(right_eye_distances_index)

            self.eye_points['right_eye'] = [point for point in right_eye_points]
            self.eye_points['left_eye'] = [point for point in left_eye_points]
            self.eye_points['left_eye_distances'] = [point for point in left_eye_distances_points]
            self.eye_points['right_eye_distances'] = [point for point in right_eye_distances_points]

            right_eyebrow_1x, right_eyebrow_1y = face_points[386][1:]
            right_eyebrow_2x, right_eyebrow_2y = face_points[374][1:]
            right_eyebrow_3x, right_eyebrow_3y = face_points[257][1:]
            right_eyebrow_4x, right_eyebrow_4y = face_points[450][1:]

            cv2.circle(face_image, (right_eyebrow_1x, right_eyebrow_1y), 4, (255, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_2x, right_eyebrow_2y), 4, (255, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_3x, right_eyebrow_3y), 4, (255, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_4x, right_eyebrow_4y), 4, (255, 0, 255), -1)


        else:
            raise Exception(f'face mesh points len: {len(face_points)} != 478')
        return self.eye_points


    def extract_eye_brows_points(self, face_points: list, face_image: np.ndarray) -> dict:  #Extrae los puntos de las cejas
        self.eyebrow_points = {'right_eyebrow': [], 'left_eyebrow': [], 'distance_between_eyebrows': [], 'distance_right_eyebrow_eye': [], 'distance_left_eyebrow_eye': [], 'distance_left_forehead':[], 'distance_right_forehead':[]}
        if len(face_points) == 478:

            right_eyebrow_index = [46, 53, 52, 65, 55]
            left_eyebrow_index = [276, 283, 282, 295, 285]
            distance_between_eyebrows_index = [55, 285]
            distance_left_forehead_index = [296, 299]
            distance_right_forehead_index = [66, 69]
            right_eye_index = [65, 468]
            left_eye_index = [295, 473]

            def get_eye_brow_points(index):
                return [face_points[i][1:] for i in index]


            right_eyebrow_points = get_eye_brow_points(right_eyebrow_index)
            left_eyebrow_points = get_eye_brow_points(left_eyebrow_index)
            distance_between_eyebrows_points = get_eye_brow_points(distance_between_eyebrows_index)
            distance_left_forehead_points = get_eye_brow_points(distance_left_forehead_index)
            distance_right_forehead_points = get_eye_brow_points(distance_right_forehead_index)
            right_eye_points = get_eye_brow_points(right_eye_index)
            left_eye_points = get_eye_brow_points(left_eye_index)

            self.eyebrow_points['right_eyebrow'] = [point for point in right_eyebrow_points]
            self.eyebrow_points['left_eyebrow'] = [point for point in left_eyebrow_points]
            self.eyebrow_points['distance_between_eyebrows'] = [point for point in distance_between_eyebrows_points]
            self.eyebrow_points['distance_left_forehead'] = [point for point in distance_left_forehead_points]
            self.eyebrow_points['distance_right_forehead'] = [point for point in distance_right_forehead_points]
            self.eyebrow_points['distance_right_eyebrow_eye'] = [point for point in right_eye_points]
            self.eyebrow_points['distance_left_eyebrow_eye'] = [point for point in left_eye_points]

            right_eyebrow_1x, right_eyebrow_1y = face_points[46][1:]
            right_eyebrow_2x, right_eyebrow_2y = face_points[53][1:]
            right_eyebrow_3x, right_eyebrow_3y = face_points[52][1:]
            right_eyebrow_4x, right_eyebrow_4y = face_points[65][1:]
            right_eyebrow_5x, right_eyebrow_5y = face_points[55][1:]

            left_eyebrow_1x, left_eyebrow_1y = face_points[276][1:]
            left_eyebrow_2x, left_eyebrow_2y = face_points[283][1:]
            left_eyebrow_3x, left_eyebrow_3y = face_points[282][1:]
            left_eyebrow_4x, left_eyebrow_4y = face_points[295][1:]
            left_eyebrow_5x, left_eyebrow_5y = face_points[285][1:]


            #Visualización de los puntos BORRAR FACE IMAGE
            cv2.circle(face_image, (right_eyebrow_1x, right_eyebrow_1y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_2x, right_eyebrow_2y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_3x, right_eyebrow_3y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_4x, right_eyebrow_4y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (right_eyebrow_5x, right_eyebrow_5y), 4, (0, 0, 255), -1)

            cv2.circle(face_image, (left_eyebrow_1x, left_eyebrow_1y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (left_eyebrow_2x, left_eyebrow_2y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (left_eyebrow_3x, left_eyebrow_3y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (left_eyebrow_4x, left_eyebrow_4y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (left_eyebrow_5x, left_eyebrow_5y), 4, (0, 0, 255), -1)

        else:
            raise Exception(f'face mesh points len: {len(face_points)} != 478')

        return self.eyebrow_points



    def extract_mouth_points(self, face_points: list, face_image: np.ndarray) -> dict:  #Extrae los puntos de la boca
        self.mouth_points = {'upper_lip': [],'lower_lip': [], 'distances': []}

        if len(face_points) == 478:

            upper_lip_index = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
            lower_lip_index = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
            mouth_opening_distance_index = [13, 14, 17, 200]

            def get_lips_points(index):
                return [face_points[i][1:] for i in index]
            
            upper_lip_points = get_lips_points(upper_lip_index)
            lower_lip_points = get_lips_points(lower_lip_index)
            mouth_opening_distance_points = get_lips_points(mouth_opening_distance_index)


            self.mouth_points['upper_lip'] = [point for point in upper_lip_points]
            self.mouth_points['lower_lip'] = [point for point in lower_lip_points]
            self.mouth_points['distances'] = [point for point in mouth_opening_distance_points]


            #upper mouth contorn
            mouth_1x, mouth_1y = face_points[78][1:]
            mouth_2x, mouth_2y = face_points[80][1:]
            mouth_3x, mouth_3y = face_points[13][1:]
            mouth_9x, mouth_9y = face_points[310][1:]
            mouth_10x, mouth_10y = face_points[308][1:]

            #lower mouth contorn
            mouth_11x, mouth_11y= face_points[95][1:]
            mouth_12x, mouth_12y = face_points[178][1:]
            mouth_13x, mouth_13y = face_points[14][1:]
            mouth_14x, mouth_14y= face_points[402][1:]
            mouth_15x, mouth_15y = face_points[324][1:]

            mouth_reference_1x, mouth_reference_1y = face_points[17][1:]
            mouth_reference_2x, mouth_reference_2y = face_points[200][1:]
            mouth_opening_1x, mouth_opening_1y = face_points[13][1:]
            mouth_opening_2x, mouth_opening_2y = face_points[14][1:]

            cv2.circle(face_image, (mouth_1x, mouth_1y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_2x, mouth_2y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_3x, mouth_3y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_9x, mouth_9y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_10x, mouth_10y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_11x, mouth_11y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_12x, mouth_12y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_13x, mouth_13y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_14x, mouth_14y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_15x, mouth_15y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (mouth_reference_1x, mouth_reference_1y), 4, (128, 0, 128), -1)
            cv2.circle(face_image, (mouth_reference_2x, mouth_reference_2y), 4, (128, 0, 128), -1)
            cv2.circle(face_image, (mouth_opening_1x, mouth_opening_1y), 4, (0, 100, 128), -1)
            cv2.circle(face_image, (mouth_opening_2x, mouth_opening_2y), 4, (0, 100, 128), -1)


        else:
            raise Exception(f'face mesh points len: {len(face_points)} != 478')
        
        return self.mouth_points

    def extract_nose_points(self, face_points: list, face_image: np.ndarray) -> dict:  #Extrae los puntos de la nariz
        self.nose_points = {'left_side': [],'right_side': [], 'reference': []}
        if len(face_points) == 478:

            nose_distances_index = [0, 13, 2, 164]

            def get_nose_points(index):
                return [face_points[i][1:] for i in index]
            
            nose_distances_points = get_nose_points(nose_distances_index)

            self.nose_points['distances'] = [point for point in nose_distances_points]

            #left side nose
            nose_1x, nose_1y = face_points[0][1:]
            nose_2x, nose_2y = face_points[13][1:]
            nose_3x, nose_3y = face_points[2][1:]
            nose_4x, nose_4y = face_points[164][1:]


            cv2.circle(face_image, (nose_1x, nose_1y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (nose_2x, nose_2y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (nose_3x, nose_3y), 4, (0, 0, 255), -1)
            cv2.circle(face_image, (nose_4x, nose_4y), 4, (0, 0, 255), -1)
        
        else:
            raise Exception(f'face mesh points len: {len(face_points)} != 478')
        
        return self.nose_points




    def main_process(self, face_image: np.ndarray, draw:bool) -> Tuple[dict, dict, dict, dict, bool, np.ndarray]:  #Procesa la imagen y devuelve los puntos de los ojos, cejas, boca y nariz
        original_image = face_image.copy()  #Copia de la imagen original
        fame_mesh_check, face_mesh_info = self.face_mesh_inference(face_image)  #Detección de puntos de la cara

        if fame_mesh_check is False:
            self.__init__();
            return self.eye_points, self.eyebrow_points, self.mouth_points, self.nose_points, False, original_image
        else:
            mesh_points = self.extract_face_mesh_points(face_image, face_mesh_info)
            if draw:
                self.draw_face_mesh(face_image, face_mesh_info, (255, 255, 0))
            self.extract_eye_points(mesh_points, face_image)
            self.extract_eye_brows_points(mesh_points, face_image)
            self.extract_mouth_points(mesh_points, face_image)
            self.extract_nose_points(mesh_points, face_image)
            return self.eye_points, self.eyebrow_points, self.mouth_points, self.nose_points, True, original_image  