import dlib
import json
import tempfile
from cog import BasePredictor, Input
from cog import Path as CogPath

class Predictor(BasePredictor):

    def setup(self):
        self.detector = dlib.get_frontal_face_detector()

    def predict(self, image: CogPath = Input(description="Path to the image")) -> CogPath:
        img = dlib.load_rgb_image(str(image))
        dets = self.detector(img)
        faces = [{"x": d.left(), "y": d.top(), "size": d.width()} for d in dets]

        out_path = CogPath(tempfile.mkdtemp()) / 'output.json'
        with open(out_path, 'w') as f:
            json.dump(faces, f)

        return out_path
