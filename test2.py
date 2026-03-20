from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input ## Change: Added preprocess_input
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    ## Change: Using the official preprocess_input function for consistency with training.
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature

## Change: Made the word lookup function highly efficient.
# It now uses a pre-built dictionary for instant lookups instead of iterating every time.
def word_for_id(integer, index_to_word_map):
    return index_to_word_map.get(integer, None)

def generate_desc(model, tokenizer, photo, max_length, index_to_word_map):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, index_to_word_map)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    # Tidy up the output
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

## Change: Loading max_length from the config file to ensure it matches the trained model.
config = load(open("config.p", "rb"))
max_length = config['max_length']

tokenizer = load(open("tokenizer.p", "rb"))
## Change: Created the efficient index-to-word map once.
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

## Change: Loading the complete model directly. This is safer and removes code duplication.
# The 'define_model' function is no longer needed in this file.
# The path is corrected to 'models2/'. Make sure to use the model you want, e.g., model_9.h5
model = load_model('models2/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length, index_to_word)
print("\n>> Generated Caption:")
print(description)
print()
plt.imshow(img)
plt.axis('off') # Hide axes
plt.show()