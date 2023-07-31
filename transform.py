from torchvision import transforms
import os


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize(256),           # Resize the input PIL Image to the given size.
    transforms.CenterCrop(224),       # Crops the given PIL Image at the center.
    transforms.ToTensor(),            # Convert the PIL Image or numpy.ndarray to tensor.
    transforms.Normalize(             # Normalize a tensor image with mean and standard deviation.
        mean=[0.485, 0.456, 0.406],   # These are the mean values for each channel for normalization.
        std=[0.229, 0.224, 0.225]     # These are the std dev values for each channel for normalization.
    )
])

# Then to apply these transformations to an image:
from PIL import Image

image = Image.open('path_to_image')
image = transform(image)

output_dir = 'path_to_output'
os.makedirs(output_dir, exist_ok=True)

# Convert tensor to PIL Image
pil_image = transforms.ToPILImage()(image).convert("RGB")

# Save the image
pil_image.save(os.path.join(output_dir, 'transformed_3.jpg'))
