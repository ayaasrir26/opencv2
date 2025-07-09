import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from typing import Tuple, Dict, List

class CheatingDetector:
    """
    Système de détection de comportements suspects en temps réel utilisant YOLOv8
    pour la surveillance d'examens.
    """
    
    def __init__(self):
        # Configuration des paramètres
        self.MIN_DISTANCE = 200  # Distance minimale d'alerte (pixels)
        self.LOOK_DURATION = 4    # Nombre de frames pour déclencher une alerte
        self.ALERT_COLOR = (0, 0, 255)  # Couleur rouge pour les alertes
        self.DISTANCE_COLOR_ALERT = (0, 0, 255)  # Rouge si distance critique
        self.DISTANCE_COLOR_SAFE = (0, 255, 0)   # Vert si distance normale
        
        # Initialisation du modèle
        self.model = YOLO('yolov8n.pt')  # Modèle pré-entraîné COCO
        
        # Structures de suivi
        self.track_history = defaultdict(list)
        self.looking_at = defaultdict(int)
        self.alert_status = defaultdict(bool)
        
    def calculate_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calcule la distance euclidienne entre les centres de deux boîtes englobantes.
        
        Args:
            box1: Boîte englobante [x, y, w, h]
            box2: Boîte englobante [x, y, w, h]
            
        Returns:
            Distance en pixels entre les centres des boîtes
        """
        center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
        center2 = (box2[0] + box2[2]/2, box2[1] + box2[3]/2)
        return np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    
    def draw_alert(self, frame: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
        """
        Dessine une alerte textuelle sur le frame avec un style professionnel.
        
        Args:
            frame: Image numpy sur laquelle dessiner
            text: Texte de l'alerte
            position: Position (x, y) du texte
            
        Returns:
            Frame avec l'alerte ajoutée
        """
        # Rectangle de fond pour une meilleure lisibilité
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_size[1] - 5),
                     (position[0] + text_size[0] + 5, position[1] + 5),
                     (50, 50, 50), -1)
        
        # Texte de l'alerte
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, self.ALERT_COLOR, 2, cv2.LINE_AA)
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Traite un frame pour détecter les comportements suspects.
        
        Args:
            frame: Image d'entrée en format numpy array
            
        Returns:
            Frame annoté avec les détections et alertes
        """
        # Détection et suivi avec YOLOv8 (classe 0 = personne)
        results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        
        # Vérification des détections
        if not results or results[0].boxes.id is None:
            return frame
            
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Analyse des interactions entre paires d'élèves
        for i, (box1, id1) in enumerate(zip(boxes, track_ids)):
            for j, (box2, id2) in enumerate(zip(boxes, track_ids)):
                if i >= j:  # Éviter les comparaisons redondantes
                    continue
                    
                distance = self.calculate_distance(box1, box2)
                
                # Détection de proximité suspecte
                if distance < self.MIN_DISTANCE:
                    self.looking_at[id1] += 1
                    self.looking_at[id2] += 1
                    
                    # Vérification des conditions d'alerte
                    if self.looking_at[id1] > self.LOOK_DURATION and not self.alert_status[id1]:
                        self.alert_status[id1] = True
                        frame = self.draw_alert(
                            frame, 
                            f"ALERTE: Élève {id1} - Comportement suspect", 
                            (30, 30 + id1 * 40)
                        )
                        
                    if self.looking_at[id2] > self.LOOK_DURATION and not self.alert_status[id2]:
                        self.alert_status[id2] = True
                        frame = self.draw_alert(
                            frame, 
                            f"ALERTE: Élève {id2} - Comportement suspect", 
                            (30, 30 + id2 * 40)
                        )
                else:
                    # Décrémenter les compteurs si la distance est sûre
                    self.looking_at[id1] = max(0, self.looking_at[id1] - 1)
                    self.looking_at[id2] = max(0, self.looking_at[id2] - 1)
                
                # Affichage des distances entre paires
                midpoint = ((box1[0] + box2[0])/2, (box1[1] + box2[1])/2)
                color = self.DISTANCE_COLOR_ALERT if distance < self.MIN_DISTANCE else self.DISTANCE_COLOR_SAFE
                cv2.putText(
                    frame, f"{distance:.0f}px", 
                    (int(midpoint[0]), int(midpoint[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
        
        # Frame annoté par YOLOv8
        return results[0].plot(img=frame)

def main():
    """Fonction principale pour exécuter la détection en temps réel."""
    detector = CheatingDetector()
    
    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(0)  # Utiliser 0 pour webcam ou chemin vers fichier vidéo
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le flux vidéo")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Avertissement: Problème de lecture du frame")
                break
            
            # Traitement et affichage
            processed_frame = detector.process_frame(frame)
            cv2.imshow("Surveillance d'Examen - Détection de Triche", processed_frame)
            
            # Sortie avec la touche 'q' ou ESC
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 27 = ESC
                break
                
    finally:
        # Nettoyage des ressources
        cap.release()
        cv2.destroyAllWindows()
        print("Système arrêté proprement")

if __name__ == "__main__":
    main()