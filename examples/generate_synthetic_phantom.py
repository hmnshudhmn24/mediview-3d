import os, numpy as np
from PIL import Image
os.makedirs("examples/synthetic_phantom", exist_ok=True)
z = 64
y = 128
x = 128
for i in range(z):
    Y, X = np.meshgrid(np.linspace(-1,1,x), np.linspace(-1,1,y))
    cx = 0.2*np.sin(2*np.pi*i/z)
    cy = 0.2*np.cos(2*np.pi*i/z)
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    img = (np.exp(-(r**2)/(2*(0.2**2))) * 255).astype(np.uint8)
    img = img + np.random.randint(0,20,size=img.shape).astype(np.uint8)
    Image.fromarray(img).save(f"examples/synthetic_phantom/slice_{i:03d}.png")
print("Synthetic phantom saved to examples/synthetic_phantom/")
