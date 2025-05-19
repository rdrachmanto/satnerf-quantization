from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from concurrent.futures import ProcessPoolExecutor
import concurrent

def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig = signal.size
    symset = list(set(signal))
    numsym = len(symset)
    propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]

    ent = np.sum([p * np.log2(1.0 / p) for p in propab])

    return ent


def read_img(path):
    colorIm = Image.open(path)
    greyIm  = colorIm.convert('L')
    colorIm = np.array(colorIm)
    greyIm  = np.array(greyIm)
    return colorIm, greyIm


def process_image(img, d, N, ent_dir):
    print("Processing ", img)
    colorIm, greyIm = read_img(os.path.join(d, img))
    S = greyIm.shape
    E = np.zeros_like(greyIm)
    for row in range(S[0]):
        for col in range(S[1]):
            Lx = max(0, col - N)
            Ux = min(S[1], col + N + 1)
            Ly = max(0, row - N)
            Uy = min(S[0], row + N + 1)
            region = greyIm[Ly:Uy, Lx:Ux].flatten()
            E[row, col] = entropy(region)

    # Assuming you want to save the entropy matrix E or return it for further processing
    np.save(os.path.join(ent_dir, f"entropy_{img}"), E)

    return np.mean(E)


def process(parent, aoi, N, ent_dir):
    d = os.path.join(parent, aoi, "ba", "crops")
    mean_ents = []
    # This will use as many processes as your machine has CPUs
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, img, d, N, ent_dir): img for img in os.listdir(d)
        }
        for future in concurrent.futures.as_completed(futures):
            img = futures[future]
            try:
                e = future.result()
                mean_ents.append(e)
                # print(f"{img} mean entropy: {e}")
            except Exception as exc:
                print(f"{img} generated an exception: {exc}")
    print(f"{aoi} mean entropy: {np.mean(mean_ents)}")


parent = os.path.abspath("Track3-preprocess")
N = 5

for aoi in os.listdir(parent):
    ent_dir = os.path.join("entropies", aoi)
    os.makedirs(ent_dir, exist_ok=True)
    process(parent, aoi, N, ent_dir)
