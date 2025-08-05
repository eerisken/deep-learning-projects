from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='tr')
result = ocr.predict('migros.jpg')

rec = result[0]  # assuming only one image

texts = rec['rec_texts']
scores = rec['rec_scores']

for txt, score in zip(texts, scores):
    print(f"{score:.2f}: {txt}")

