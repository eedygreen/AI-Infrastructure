
from PIL import Image
from typing import Tuple, List
import numpy as np
import random

class Compose:
   """
        Compose several transforms together

   """
   def __init__(self, transforms: List):
      self.transforms = transforms
      
   def __call__(self, img: Image.Image) -> np.ndarray:
      for transform in self.transforms:
        img = transform(img)
      return img

class RandomRotation:
   """
        Rotate the image by a random angle within any given range
   """
   def __init__(self, degrees: float):
      """
        Args:\n
            degrees: Range of degrees (30 means -30 to +30)
      """
      self.degrees = degrees

   def __call__(self, img: Image.Image) -> Image.Image:
      angle = random.uniform(-self.degrees, self.degrees)
      return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))
   
class RandomResizedCrop:
   """
        Crop the image to random size, aspect ratio and resize to target size
   """
   def __init__(
         self, 
         size: int, 
         scale: Tuple[float, float] = (0.08, 1.0),
         ratio: Tuple[float, float] = (3./4., 3./4.)
    ):
    """
        Args: \n
            size: target output size (will be size x size) \n
            scale: Range of size \n
            ratio: Range of aspect ratio
    """
    self.size = (size, size) if isinstance(size, int) else size
    self.scale = scale
    self.ratio = ratio

   def __call__(self, img: Image.Image) -> Image.Image:
      width, height = img.size
      area = width * height
      
      for _ in range(10):
         target_area = random.uniform(*self.scale) * area
         aspect_ratio = random.uniform(*self.ratio)

         w = int(round((target_area * aspect_ratio) ** 0.5))
         h = int(round((target_area / aspect_ratio) ** 0.5))

         if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            img = img.crop((j, i, j + w, i + h))
            return img.resize(self.size, Image.BILINEAR)
         
      in_ratio = width / height
      if in_ratio < min(self.ratio):
         w = width
         h = int(round(w / min(self.ratio)))
      elif in_ratio > max(self.ratio):
         h = height
         w = int(random(h * max(self.ratio)))
      else:
         w = width
         h = height

      i = (height - h) // 2
      j = (width - w ) // 2
      img = img.crop((j, i, j + w, i + h))
      return img.resize(self.size, Image.BILINEAR) 
          

class RandomHorizontalFlip:
   """
        Horizontally flip the image randomly with a given probability
   """
   def __init__(self, p: float = 0.5):
      """
        Args:\n
            p: Probability of the image being flipped (default: 0.5, i.e 50%)
      """
      self.p = p

   def __call__(self, img: Image.Image) -> Image.Image:
      if random.random() < self.p:
         return Image.mirror(img)
      return img
   
class Resize:
   """
        Resize the Image to any given size
   """
   def __init__(self, size: int):
      """
        Args: \n
            size: Desired output size. If size is an Int, smaller edge of the image will be matched to this number.
      """
      self.size = size

   def __call__(self, img: Image.Image) -> Image.Image:
      width, height = img.size

      if width < height:
         new_width = self.size
         new_height = int(self.size * height / width)
      else:
         new_height = self.size
         new_width = int(self.size * width / height)
      return img.resize((new_width, new_height), Image.BILINEAR)
   
class CenterCrop:
   """
    Crop the image at the center to the given size
   """
   def __init__(self, size: int):
      """
        Args:\n
            size: Desired output size (will be size x size)
      """
      self.size = (size, size) if isinstance(size, int) else size

   def __call__(self, img: Image.Image) -> Image.Image:
      width, height = img.size
      crop_height, crop_width = self.size

      left = (width - crop_width) // 2
      top = (height - crop_height) // 2
      right = left + crop_width
      bottom = top + crop_height

      return img.crop((left, top, right, bottom))
   
class ToTensor:
   """
        Convert a PIL Image to a numpy array tensor in CHW format with values in [0, 1]
   """
   def __call__(self, img: Image.Image) -> np.ndarray:
      # convert PIL image to numpy array
      img_array = np.array(img, dtype=np.float32)
      
      # Handle grayscale images
      if len(img_array.shape) == 2:
         img_array = np.stack([img_array] * 3, axis=-1)

      # normalise to [0, 1]
      img_array = img_array / 255.0

      # Convert from HWC to CWH format (Horizontal, Width, Channel -> Channles, Height, Width)
      img_array = np.transpose(img_array, (2, 0, 1))

      return img_array
   
class Normalize:
   """
        Normalize a tensor image with mean and standard deviation
   """
   def __init__(self, mean: List[float], std: List[float]):
      """
        Args: \n
            mean: Sequence of means for each channel
            std: Sequence of standard deviations for each channel
      """
      self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
      self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)
      
   def __call__(self, tensor: np.ndarray) -> np.ndarray:
      # Normalize: (image - mean) / std
      return (tensor - self.mean) / self.std
