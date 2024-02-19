import os
from PIL import Image

import cohere
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

COLOURS = [
    "violet",
    "blue",
    "cyan",
    "green",
    "yellow",
    "orange",
    "red",
]

def rotate_vector(vector, theta):
    return vector @ np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def concatenate_images():
    ROOT_DIR = "emb3d_imgs"
    paths = sorted([(os.path.join(ROOT_DIR, path)) for path in os.listdir(ROOT_DIR)])
    images = [Image.open(path) for path in paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    concatenated_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        concatenated_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    concatenated_image.save(os.path.join(ROOT_DIR, 'emb3d_concatenated.png'))

def get_individual_pngs():
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    embs = co.embed(model='embed-english-v3.0', texts=COLOURS, input_type='classification').embeddings
    embedding_dim = len(embs[0])
    xaxis = np.array(list(range(embedding_dim)))-embedding_dim//2

    for scale in [1, 100, 1000, 2000, 4000]:
        fig, ax = plt.subplots(figsize=(20, 20))
        for i, (emb, colour) in enumerate(zip(embs, COLOURS)):
            # Create a plot
            emb = np.array(emb)*scale
            vec = np.array([xaxis, emb]).T
            rotated = rotate_vector(vec, theta=(i+1)*2*np.pi/len(COLOURS)).T
            ax.plot(rotated[0], rotated[1], color=colour, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.savefig(f'emb3d_imgs/emb3d_{scale}.png')

if __name__ == '__main__':
    get_individual_pngs()
    concatenate_images()