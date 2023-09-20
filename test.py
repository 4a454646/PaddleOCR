import cv2
import imgaug.augmenters as iaa

# Load the input image
image = cv2.imread('/workspace/PaddleOCR/aug_train/cards_crops/headings/0_0fe8fb453246fb076c3c278779978384feb5dd57.jpg')

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25), fit_output=True),
], random_order=True)

# Apply the augmentation sequence to the image
augmented_image = seq.augment_image(image)

# Save the augmented image to the visualizations folder
cv2.imwrite('/workspace/PaddleOCR/visualize/augmented_image.jpg', augmented_image)