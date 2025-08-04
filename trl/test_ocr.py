from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    lang="en",
    use_doc_unwarping=False,
    use_textline_orientation=False)

input_image = Image.open("test_ocr.png")
img_bgr = np.array(input_image)[:, :, ::-1]


# .ocr() is an alias for .predict(); we'll get a list of [ [boxes, rec_texts, rec_scores], … ] per line
result = ocr.ocr([img_bgr])
print(result[0]['rec_texts'])

