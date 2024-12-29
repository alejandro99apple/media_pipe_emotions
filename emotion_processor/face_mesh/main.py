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
        self.eyebrown_points: dict = {}
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
        



    def extract_face_mesh_points(self, face_image:np.ndarray, face_mesh_info:Any, viz:bool):  #Extrae los puntos de los ojos, cejas, boca y nariz
        h, w, c = face_image.shape  #Dimensiones de la imagen
        self.mesh_points = []
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for i, point in enumerate(face_mesh.landmark):
                x, y = int(point.x*w), int(point.y*h)   
                self.mesh_points.append([i, x, y])  #Guarda los puntos de la cara

            if viz is True:
                self.mp_draw.draw_landmarks(face_image, face_mesh, self.mp_face_mesh_object.FACEMESH_TESSELATION, self.config_draw, self.config_draw)  #Dibuja los puntos de la cara

        return self.mesh_points




    def main_process(self, face_image: np.ndarray) -> Tuple[dict, dict, dict, dict, str, np.ndarray]:  #Procesa la imagen y devuelve los puntos de los ojos, cejas, boca y nariz
        original_image = face_image.copy()  #Copia de la imagen original
        fame_mesh_check, face_mesh_info = self.face_mesh_inference(face_image)  #Detección de puntos de la cara

        if fame_mesh_check is False:
            return self.eye_points, self.eyebrown_points, self.mouth_points, self.nose_points, 'No face detected', original_image
        else:
            mesh_points = self.extract_face_mesh_points(face_image, face_mesh_info, viz=True)
            return self.eye_points, self.eyebrown_points, self.mouth_points, self.nose_points, 'Face detected', original_image