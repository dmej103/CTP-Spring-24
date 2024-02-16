from transformers import pipeline
from PIL import Image


# Used emotion classifier model
classifier = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions")
print(classifier("I am excited to build apps with AI.")) # It succesfully classified this as 'excited'

# Used image-classifier model on a locally stored image
img = Image.open("/Users/david/Documents/CunyTechPrep/Spring-24/HW-1/savanna.jpg")
image_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
print(image_classifier(img)) # Pretty funny, it classified this nature photo of the savannah as NSFW


'''
I had to specify a model or else I would receive an error that said:
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
'''