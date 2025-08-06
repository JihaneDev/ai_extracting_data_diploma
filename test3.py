import os
import logging
from PIL import Image
import pytesseract

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

class PDFPreprocessor:
    @staticmethod
    def auto_rotate(image: Image.Image) -> Image.Image:
        """
        Fait pivoter automatiquement l'image si le texte n'est pas horizontal.
        Utilise Tesseract pour détecter l'orientation.
        """
        try:
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            angle = osd.get("rotate", 0)
            if angle != 0:
                logger.info(f"Rotation automatique de {angle}°")
                return image.rotate(-angle, expand=True)
            return image
        except Exception as e:
            logger.warning(f"Échec de la rotation automatique : {str(e)}")
            return image

# === Exemple d'utilisation ===
if __name__ == "__main__":
    # Remplace ce chemin par une image scannée contenant du texte
    image_path = r"C:\Users\hp\Downloads\5_DIPLOME.pdf"
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        rotated_img = PDFPreprocessor.auto_rotate(img)
        rotated_img.show()  # Affiche l'image corrigée
    else:
        logger.error(f"L'image n'existe pas : {image_path}")