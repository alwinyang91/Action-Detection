import cv2

def mediapipe_detection(image, model):
    # Convert the image to RGB color space.
    # Reason: The Mediapipe library expects input images to be in RGB format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Set the image as read-only to improve performance and prevent accidental modifications.
    # Reason: Making the image read-only can improve performance and prevent accidental modifications to the image.
    image.flags.writeable = False                 

    # Process the image using the specified Mediapipe model.
    # Reason: This is the main processing step, where the model is applied to the image to detect objects or estimate poses.
    results = model.process(image)                

    # Set the image as writeable again, in case further processing is needed.
    # Reason: This step is necessary if further processing of the image is required after the Mediapipe model has been applied.
    image.flags.writeable = True                   

    # Convert the image back to the BGR color space.
    # Reason: OpenCV uses BGR format by default, so the image must be converted back to this format before it can be displayed or saved.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    # Return the modified image and the results of the Mediapipe model.
    # Reason: This function is designed to return both the modified image and the results of the Mediapipe model, so that they can be used for further processing or display.
    return image, results



def draw_landmarks(image, results, mp_drawing, mp_holistic):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )  
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


