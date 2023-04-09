from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()
    #main function takes takes the query image path and also save the feature extacted from the image

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  
        np.save(feature_path, feature)#saves the feature along with its path
