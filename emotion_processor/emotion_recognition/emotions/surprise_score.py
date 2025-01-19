

class SurpriseScore:

    def calculate_score(self, features:dict) -> float:
        eyebrows, eye, mouth, nose  = features
        eyebrows = eyebrows.eyebrows
        eye = eye.eyes
        mouth = mouth.mouth


        eye_score = self.calculate_eyes_score(eye)
        eyebrows_score = self.calculate_eyebrows_score(eyebrows)
        mouth_score = self.calculate_mouth_score(mouth)

        score = (eye_score + eyebrows_score + mouth_score)
        #print('Sorpresa: ',score)
        return score


    def calculate_eyes_score(self, eye: str) -> float:
        score = 0.0

        left_eye_short_distance = eye['left_eye_short_distance']
        left_eye_big_distance = eye['left_eye_big_distance']
        right_eye_short_distance = eye['right_eye_short_distance']
        right_eye_big_distance = eye['right_eye_big_distance']

        if(left_eye_short_distance / left_eye_big_distance >= 0.32 ):
            #print('OJO ABIEERTO')
            score += 40
        elif(left_eye_short_distance / left_eye_big_distance >= 0.1 ):
            #print('OJO Entreabierto')
            score += 10
        else: 
            #print('OJO CERRADO')
            score += 0

        #print(score)
        return score
        

    def calculate_eyebrows_score(self, eyebrows: str) -> float:
        score = 0.0

        distance_right_eyebrow_eye = eyebrows['distance_right_eyebrow_eye']
        distance_right_forehead = eyebrows['distance_right_forehead']
        distance_left_eyebrow_eye = eyebrows['distance_left_eyebrow_eye']
        distance_left_forehead = eyebrows['distance_left_forehead']

        if distance_right_eyebrow_eye*0.40 > distance_right_forehead:
            #print('Ceja derecha levantada')
            score += 10
        else:
            #print('ceja derecha normal',)
            score += 0

        if distance_left_eyebrow_eye*0.40 > distance_left_forehead:
            #print('Ceja izquierda levantada')
            score += 10
        else:
            #print('Ceja izquierda normal')
            score += 0

        #print(score)
        return score
    


    def calculate_mouth_score(self, mouth: str) -> float:
        score = 0.0
        mouth_opening_distance = mouth['mouth_opening_distance']
        mouth_horizontal_distance = mouth['mouth_horizontal_distance']
        mouth_reference_distance = mouth['mouth_reference_distance']

        if(mouth_opening_distance / mouth_horizontal_distance >= 0.6):
            #print('BOCA ABIERTA EN O')
            score += 40
        else: 
            #print('BOCA no abierta en O')
            score += 0
        #print(score)
        return score

