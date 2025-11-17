# %% [markdown] {"id":"Y6pQyy8aujMV"}
# # Overview

# %% [markdown] {"id":"XHaTogovuA4-"}
# ## Dataset Overview

# %% [markdown] {"id":"zM6HMaxnuL3E"}
# Oxford‑IIIT Pet contains 37 pet breeds with roughly 200 images per class, and images vary significantly in scale, pose, and lighting, which makes it a realistic benchmark for robust classification.​
# In addition to images, the dataset ships with annotations including breed labels, species labels, segmentation masks, and head bounding boxes for the training split.

# %% [markdown] {"id":"yL1WrYckuqTH"}
# ## Labels and Splits

# %% [markdown] {"id":"-DdeyWJXut95"}
# TFDS exposes two official splits: 3,680 training examples and 3,669 test examples, for a total of 7,349 images.​
# The feature dictionary includes image tensors, a 37‑class “label” (breed), a two‑class “species”, optional segmentation masks, and training‑only head bounding boxes, and for this project the target will be the two‑class“species"

# %% [markdown] {"id":"j4KYCPZrvBBC"}
# ## Install and import

# %% [code] {"id":"LBqKXAGInE72","execution":{"iopub.status.busy":"2025-11-15T05:39:28.743513Z","iopub.execute_input":"2025-11-15T05:39:28.743809Z","iopub.status.idle":"2025-11-15T05:39:49.722621Z","shell.execute_reply.started":"2025-11-15T05:39:28.743787Z","shell.execute_reply":"2025-11-15T05:39:49.721924Z"}}
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import collections

# %% [code] {"id":"21Hg1A21vDce","execution":{"iopub.status.busy":"2025-11-15T05:39:49.723787Z","iopub.execute_input":"2025-11-15T05:39:49.724746Z","iopub.status.idle":"2025-11-15T05:39:49.728176Z","shell.execute_reply.started":"2025-11-15T05:39:49.724705Z","shell.execute_reply":"2025-11-15T05:39:49.727364Z"}}
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# %% [markdown] {"id":"zCigGeggvFUf"}
# - Images will be resized to 224×224 pixels (the size expected by VGG16).
# 
# - Data will be processed in batches of 32 images at a time.

# %% [code] {"id":"QBaxlvpDvPil","outputId":"1e298b06-36e5-4a12-e4b8-4301dcf69745","execution":{"iopub.status.busy":"2025-11-15T05:39:49.728743Z","iopub.execute_input":"2025-11-15T05:39:49.728921Z","iopub.status.idle":"2025-11-15T05:39:50.469722Z","shell.execute_reply.started":"2025-11-15T05:39:49.728905Z","shell.execute_reply":"2025-11-15T05:39:50.469084Z"}}
ds_info = tfds.builder('oxford_iiit_pet').info
print("Species classes:", ds_info.features['species'].names)
print("Breed classes (#37):", ds_info.features['label'].names[:5], "...")

# %% [markdown] {"id":"djD2QGPwvVO5"}
# - Loads dataset metadata without downloading images.
# 
# - The dataset has:
# 
#    - species: binary labels (['cat', 'dog'])
# 
#    - label: 37 breed categories (used for multi-class tasks).
# 
# - Printing confirms that you’re using the binary species feature.

# %% [code] {"id":"9Ei1g2l6vUoH","execution":{"iopub.status.busy":"2025-11-15T05:39:50.471411Z","iopub.execute_input":"2025-11-15T05:39:50.471626Z","iopub.status.idle":"2025-11-15T05:39:50.475695Z","shell.execute_reply.started":"2025-11-15T05:39:50.471608Z","shell.execute_reply":"2025-11-15T05:39:50.474955Z"}}
def preprocess(example):
    image = tf.image.resize(example['image'], IMG_SIZE)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    label = tf.cast(example['species'], tf.int32)
    return image, label

# %% [markdown] {"id":"iv_j-8qavy4c"}
# - Resizes each image to (224, 224).
# 
# - Preprocesses it using VGG16’s standard input format:
# 
#    - Converts RGB → BGR
# 
#     - Subtracts ImageNet mean pixel values
# 
# - Extracts label: species (0 for cat, 1 for dog).
# 
# - Returns (image, label) pairs.

# %% [code] {"id":"54zJKdudvvwj","outputId":"296e90f4-be51-4596-e863-47f081ac33e6","execution":{"iopub.status.busy":"2025-11-15T05:39:50.476325Z","iopub.execute_input":"2025-11-15T05:39:50.476511Z","iopub.status.idle":"2025-11-15T05:40:55.204197Z","shell.execute_reply.started":"2025-11-15T05:39:50.476496Z","shell.execute_reply":"2025-11-15T05:40:55.203559Z"}}
splits = ['train[:85%]', 'train[85%:]', 'test']
train_raw, val_raw, test_raw = tfds.load('oxford_iiit_pet', split=splits, as_supervised=False)

# %% [markdown] {"id":"AkdjBBp9wRBF"}
# - The original dataset has a train and test split.
# 
# - From the train split:
# 
#   - First 85% → training set
# 
#   - Last 15% → validation set
# 
# - The test split → held-out final test set.
# 
# - as_supervised=False keeps the data in dictionary format (example['image'], example['species'], etc.).

# %% [code] {"id":"dgjiUuW3wN7h","execution":{"iopub.status.busy":"2025-11-15T05:40:55.204899Z","iopub.execute_input":"2025-11-15T05:40:55.205207Z","iopub.status.idle":"2025-11-15T05:40:55.369777Z","shell.execute_reply.started":"2025-11-15T05:40:55.205187Z","shell.execute_reply":"2025-11-15T05:40:55.369147Z"}}
train = (train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
         .shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
val = (val_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
       .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
test = (test_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# %% [markdown] {"id":"DGQL_z_fxDtF"}
# - .map(preprocess): applies your preprocessing function to every example.
# 
# - .shuffle(2048): randomizes training data order for better generalization.
# 
# - .batch(BATCH_SIZE): groups samples into batches of 32.
# 
# - .prefetch(tf.data.AUTOTUNE): overlaps preprocessing with training for speed.
# 
# - These create optimized TensorFlow input pipelines.

# %% [code] {"id":"Jhst0hq1w_LD","execution":{"iopub.status.busy":"2025-11-15T05:40:55.370599Z","iopub.execute_input":"2025-11-15T05:40:55.370828Z","iopub.status.idle":"2025-11-15T05:40:58.299711Z","shell.execute_reply.started":"2025-11-15T05:40:55.370810Z","shell.execute_reply":"2025-11-15T05:40:58.299062Z"}}
species_names = ds_info.features['species'].names
def show_batch(ds, n=6):
    images, labels = next(iter(ds.unbatch().batch(n)))
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(tf.keras.utils.array_to_img(images[i]))
        plt.title(species_names[int(labels[i])])
        plt.axis('off')
    plt.show()

# %% [code] {"id":"wOeZBDOL2MCj","outputId":"ae21c204-7df5-4af1-de13-4ab6ffbd45dc","execution":{"iopub.status.busy":"2025-11-15T05:40:58.300582Z","iopub.execute_input":"2025-11-15T05:40:58.300859Z","iopub.status.idle":"2025-11-15T05:41:02.805587Z","shell.execute_reply.started":"2025-11-15T05:40:58.300828Z","shell.execute_reply":"2025-11-15T05:41:02.804612Z"}}
show_batch(train)

# %% [code] {"id":"VRM0oCPk2R4Q","outputId":"ce4e2428-79ed-4cbb-c532-bf72403ed2e4","execution":{"iopub.status.busy":"2025-11-15T05:41:02.806631Z","iopub.execute_input":"2025-11-15T05:41:02.806847Z","iopub.status.idle":"2025-11-15T05:41:09.211195Z","shell.execute_reply.started":"2025-11-15T05:41:02.806829Z","shell.execute_reply":"2025-11-15T05:41:09.210313Z"}}
import collections
counter = collections.Counter()
for _, y in train.unbatch():  # sample to be quick
    counter[int(y)] += 1
print("Sampled species counts:", counter, "->", {species_names[k]: v for k, v in counter.items()})

# %% [markdown] {"id":"6FSerwEp4T_l"}
# ## Label Counts

# %% [code] {"id":"qA17lJ3B3fZU","outputId":"bc5dc25f-bfcb-46fa-caf9-6c7e7c60fb95","execution":{"iopub.status.busy":"2025-11-15T05:41:09.214253Z","iopub.execute_input":"2025-11-15T05:41:09.214689Z","iopub.status.idle":"2025-11-15T05:41:23.444865Z","shell.execute_reply.started":"2025-11-15T05:41:09.214669Z","shell.execute_reply":"2025-11-15T05:41:23.444251Z"}}
def count_classes(ds, name):
    counter = collections.Counter()
    for _, y in ds.unbatch():
        counter[int(y)] += 1
    print(f"{name} set:", {species_names[k].capitalize(): v for k, v in counter.items()})
    return counter  # <-- return added

train_counts = count_classes(train, "Train")
val_counts = count_classes(val, "Validation")
test_counts = count_classes(test, "Test")

species_names = ds_info.features['species'].names

# Extract counts
splits = ['Train', 'Validation', 'Test']
cat_counts = [train_counts.get(0, 0), val_counts.get(0, 0), test_counts.get(0, 0)]
dog_counts = [train_counts.get(1, 0), val_counts.get(1, 0), test_counts.get(1, 0)]

# Plot
x = range(len(splits))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar([i - width/2 for i in x], cat_counts, width, label='Cat')
plt.bar([i + width/2 for i in x], dog_counts, width, label='Dog')
plt.xticks(x, splits)
plt.ylabel('Number of Images')
plt.title('Class Distribution by Split')
plt.legend()
plt.show()

# %% [markdown] {"id":"emhIpywQCbur"}
# ## Why VGG16 and its preprocessing

# %% [markdown] {"id":"yeYmJc7qCgpL"}
# VGG16 is a classic ImageNet‑pretrained model that expects 224×224 RGB inputs and specific channel preprocessing via vgg16.preprocess_input, making it simple and reliable for transfer learning on small to medium datasets.​
# The standard transfer learning workflow is to freeze the pretrained convolutional base, train a lightweight classification head, then unfreeze the upper convolution blocks for low‑LR fine‑tuning to adapt features to the task.

# %% [code] {"id":"vCYS5CIJM1mP","execution":{"iopub.status.busy":"2025-11-15T05:41:23.445671Z","iopub.execute_input":"2025-11-15T05:41:23.445881Z","iopub.status.idle":"2025-11-15T05:41:23.455189Z","shell.execute_reply.started":"2025-11-15T05:41:23.445865Z","shell.execute_reply":"2025-11-15T05:41:23.454597Z"}}
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# %% [code] {"id":"gnUq7MTf4o4v","outputId":"55121126-39d3-4e19-cc27-d5070a6f32f0","execution":{"iopub.status.busy":"2025-11-15T05:41:23.456033Z","iopub.execute_input":"2025-11-15T05:41:23.456309Z","iopub.status.idle":"2025-11-15T05:41:25.149069Z","shell.execute_reply.started":"2025-11-15T05:41:23.456284Z","shell.execute_reply":"2025-11-15T05:41:25.148211Z"}}
base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base.trainable = False

# %% [markdown] {"id":"TMR-t2amMhwS"}
# - Loads VGG16 pretrained on ImageNet.
# 
# - include_top=False → excludes the final classifier layers, leaving only convolutional feature extractor.
# 
# - base.trainable = False → freeze the weights so training doesn’t modify pretrained features initially.

# %% [code] {"id":"cYzqcPpzL-gU","execution":{"iopub.status.busy":"2025-11-15T05:41:25.149977Z","iopub.execute_input":"2025-11-15T05:41:25.150277Z","iopub.status.idle":"2025-11-15T05:41:25.168849Z","shell.execute_reply.started":"2025-11-15T05:41:25.150252Z","shell.execute_reply":"2025-11-15T05:41:25.167980Z"}}
data_aug = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# %% [markdown] {"id":"M8mOco5rY49I"}
# - Adds data augmentation to reduce overfitting:
# 
#   - Random horizontal flip
# 
#   - Random rotation (up to ±10%)
# 
#   - Random zoom

# %% [code] {"id":"VInjfKqkMsFk","execution":{"iopub.status.busy":"2025-11-15T05:41:25.169611Z","iopub.execute_input":"2025-11-15T05:41:25.169783Z","iopub.status.idle":"2025-11-15T05:41:25.997868Z","shell.execute_reply.started":"2025-11-15T05:41:25.169770Z","shell.execute_reply":"2025-11-15T05:41:25.997076Z"}}
inputs = layers.Input(shape=(224,224,3))
x = data_aug(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

# %% [markdown] {"id":"nF4QRBqnZMrw"}
# - inputs → placeholder for 224×224 RGB images.
# 
# - Pass through augmentation, then VGG16 backbone.
# 
# - GlobalAveragePooling2D → converts convolutional feature maps into a 1D vector.
# 
# - Dropout(0.2) → prevents overfitting by randomly zeroing 20% of neurons.
# 
# - Dense(1, activation='sigmoid') → outputs a probability for dog vs cat.
# 
# - model → complete Keras Model object.

# %% [code] {"id":"KJY6ENAaZKI4","execution":{"iopub.status.busy":"2025-11-15T05:41:25.998720Z","iopub.execute_input":"2025-11-15T05:41:25.998987Z","iopub.status.idle":"2025-11-15T05:41:26.816766Z","shell.execute_reply.started":"2025-11-15T05:41:25.998962Z","shell.execute_reply":"2025-11-15T05:41:26.815811Z"}}
model.compile(optimizer=optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% [markdown] {"id":"GeULOpRnZ5jT"}
# - Optimizer → Adam with learning rate 0.001.
# 
# - Loss → binary crossentropy (since it’s a 2-class problem).
# 
# - Metrics → track accuracy during training.

# %% [markdown] {"id":"MvVpMFUvI3Ue"}
# ## Model Training

# %% [code] {"id":"nrOhtBAIZX_Q","outputId":"9dc029a5-573a-4684-ebfe-234b199d7ef3","execution":{"iopub.status.busy":"2025-11-15T05:41:26.817724Z","iopub.execute_input":"2025-11-15T05:41:26.817980Z","iopub.status.idle":"2025-11-15T05:45:19.377749Z","shell.execute_reply.started":"2025-11-15T05:41:26.817956Z","shell.execute_reply":"2025-11-15T05:45:19.376920Z"}}
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ckpt = tf.keras.callbacks.ModelCheckpoint('vgg16_cats_dogs.keras', monitor='val_accuracy', mode='max', save_best_only=True)

history = model.fit(train, validation_data=val, epochs=20, callbacks=[early, ckpt], verbose=1)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-15T05:45:19.378745Z","iopub.execute_input":"2025-11-15T05:45:19.379171Z","iopub.status.idle":"2025-11-15T05:45:19.663888Z","shell.execute_reply.started":"2025-11-15T05:45:19.379143Z","shell.execute_reply":"2025-11-15T05:45:19.663248Z"}}
model.save('vgg16catsdogs.keras')
print("Model saved successfully.")

# %% [markdown] {"id":"k_j37yZ_aMFv"}
# - EarlyStopping: stops training if validation loss doesn’t improve for 3 epochs.
# 
# - ModelCheckpoint: saves the model weights with the best validation accuracy.
# 
# - history stores loss and accuracy per epoch for both train and validation sets.

# %% [markdown] {"id":"3wWchqM_eDMx"}
# **Quick guidance on reading curves**
# - Healthy learning shows training and validation loss both decreasing, with validation accuracy rising and staying close to training accuracy across epochs.​
# 
# - If validation loss rises while training loss keeps falling, that indicates overfitting; consider EarlyStopping or data augmentation/fine‑tuning adjustments.

# %% [markdown] {"id":"ClPikrmwHpaN"}
# ## Model Evaluation

# %% [code] {"id":"KxYQGXRZaHq9","outputId":"07b5f2cb-6b19-40f7-f743-1baec1188330","execution":{"iopub.status.busy":"2025-11-15T05:45:19.664573Z","iopub.execute_input":"2025-11-15T05:45:19.664778Z","iopub.status.idle":"2025-11-15T05:45:19.838622Z","shell.execute_reply.started":"2025-11-15T05:45:19.664763Z","shell.execute_reply":"2025-11-15T05:45:19.838036Z"}}
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss Over Epochs')     # Add a title
plt.xlabel('Epochs')                    # Label for x-axis
plt.ylabel('Loss')                      # Label for y-axis
plt.legend()                            # Show the legend
plt.show()

# %% [code] {"id":"JKXOfCZ-epQo","outputId":"ccdd9524-df10-464a-e09a-86477be02749","execution":{"iopub.status.busy":"2025-11-15T05:45:19.839285Z","iopub.execute_input":"2025-11-15T05:45:19.839518Z","iopub.status.idle":"2025-11-15T05:45:20.034184Z","shell.execute_reply.started":"2025-11-15T05:45:19.839497Z","shell.execute_reply":"2025-11-15T05:45:20.033552Z"}}
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

# %% [code] {"id":"KqX_2zl9HGFO","outputId":"8477c866-5b72-4d7d-8389-4d6c93b08d61","execution":{"iopub.status.busy":"2025-11-15T05:45:20.034877Z","iopub.execute_input":"2025-11-15T05:45:20.035117Z","iopub.status.idle":"2025-11-15T05:45:45.180941Z","shell.execute_reply.started":"2025-11-15T05:45:20.035101Z","shell.execute_reply":"2025-11-15T05:45:45.180354Z"}}
# Load the trained model
best_model = tf.keras.models.load_model('vgg16_cats_dogs.keras')

test_loss, test_acc = best_model.evaluate(test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# %% [markdown] {"id":"cNrO4V9xI8Cd"}
# ## Model Testing

# %% [code] {"execution":{"iopub.status.busy":"2025-11-15T05:45:45.181811Z","iopub.execute_input":"2025-11-15T05:45:45.182498Z","iopub.status.idle":"2025-11-15T05:45:49.080051Z","shell.execute_reply.started":"2025-11-15T05:45:45.182472Z","shell.execute_reply":"2025-11-15T05:45:49.079273Z"}}

# %% [code] {"id":"CKOUs0qomuap","outputId":"722c4348-671d-4aef-dd97-bca70191f48d","execution":{"iopub.status.busy":"2025-11-15T05:45:49.081252Z","iopub.execute_input":"2025-11-15T05:45:49.081529Z","iopub.status.idle":"2025-11-15T05:45:50.374990Z","shell.execute_reply.started":"2025-11-15T05:45:49.081505Z","shell.execute_reply":"2025-11-15T05:45:50.373922Z"}}
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

app = FastAPI()

# Password hashing and token scheme setup (simple example)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simple user DB for example (store hashed passwords in real apps)
fake_users_db = {
    "user": {"username": "user", "hashed_password": pwd_context.hash("pass")}
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # Generate a token - here we simply use username as token for demo purposes
    return {"access_token": user['username'], "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

# Load the TensorFlow Keras model once at startup
model = tf.keras.models.load_model("vgg16catsdogs.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    preds = model.predict(img_array)
    confidence = float(preds[0][0])
    label = "Dog" if confidence > 0.5 else "Cat"
    confidence = confidence if confidence > 0.5 else 1 - confidence

    return {"label": label, "confidence": confidence}

# %% [code] {"execution":{"iopub.status.busy":"2025-11-15T05:50:29.269554Z","iopub.execute_input":"2025-11-15T05:50:29.269856Z","iopub.status.idle":"2025-11-15T05:50:29.653140Z","shell.execute_reply.started":"2025-11-15T05:50:29.269829Z","shell.execute_reply":"2025-11-15T05:50:29.652074Z"}}
import nest_asyncio
import uvicorn
from pyngrok import ngrok
from your_fastapi_module import app  # import your FastAPI app here

# Apply asyncio patch for nested event loops (needed in some environments)
nest_asyncio.apply()

# Set ngrok auth token, **without the '!'**
ngrok.set_auth_token("33EQaovSQuTua0LUU3ns0lCs20Z_6yXKoXcNrW9DdYHKw9aze")

# Open an ngrok tunnel to the FastAPI port
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run the FastAPI app on port 8000
uvicorn.run(app, host="0.0.0.0", port=8000)


# %% [code]
