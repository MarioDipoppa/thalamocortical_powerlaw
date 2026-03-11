import argparse

import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

def main(args:argparse.ArgumentParser):
    
    # for deterministic behavior
    np.random.seed(args.seed)
    
    # load the entire video into memory
    props = iio.improps(args.video)
    video = iio.imread(args.video, index=None).astype(np.float32) / 255.
    video = np.dot(video, [[0.2989], [0.5870], [0.1140]])   # convert to grayscale
    
    # compute the (x, y) top-left coords to use
    X = np.random.uniform(0, props.shape[2] - int(args.crop_size), int(args.n_triplets) // props.n_images)
    Y = np.random.uniform(0, props.shape[1] - int(args.crop_size), int(args.n_triplets) // props.n_images)
    
    # create a meshgrid to vectorize the crop generation
    offsets = np.arange(int(args.crop_size))
    dx, dy = np.meshgrid(offsets, offsets, indexing='xy')
    
    # store the set of x and y pixels per crop
    X = (X[:, None, None] + dx[None, :, :]).astype(np.uint32)
    Y = (Y[:, None, None] + dy[None, :, :]).astype(np.uint32)
    
    # set up the storage on disk
    triplets = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=(int(args.n_triplets), 3, int(args.crop_size), int(args.crop_size)))
    
    n_per_frame = int(args.n_triplets) // props.n_images

    for i,f in tqdm(enumerate(video)):

        a = f[Y, X].squeeze()

        if i + int(args.an_dist) > props.n_images - 1:
            p = video[i-int(args.ap_dist)][Y, X].squeeze()
            n = video[i-int(args.an_dist)][Y, X].squeeze()
        else:
            p = video[i+int(args.ap_dist)][Y, X].squeeze()
            n = video[i+int(args.an_dist)][Y, X].squeeze()

        triplet_set = np.stack([a,p,n],axis=1)

        start = i * n_per_frame
        end = start + n_per_frame
        triplets[start:end] = triplet_set
        for j,k in zip(range(i * (int(args.n_triplets) // props.n_images)), range(int(args.n_triplets) // props.n_images)):
            triplets[i * (int(args.n_triplets) // props.n_images) + j] = triplet_set[k]

if __name__ == "__main__":
    
    # pase the arguments the user gives and just pass them to main
    parser = argparse.ArgumentParser(description="generate triplets from a given video")
    parser.add_argument("--video", help="path to video to generate triplets from")
    parser.add_argument("--ap-dist", help="the number of frames that differentiate the anchor and positive images", default=1)
    parser.add_argument("--an-dist", help="the number of frames that differentiate the anchor and negative images", default=50)
    parser.add_argument("--n-triplets", help="the total number of triplets to generate", default=10000)
    parser.add_argument("--crop-size", help="the dimensions of each of the images in the triplet", default=224)
    parser.add_argument("--data-key", help="the key in the .h5 file where the data is stored under")
    parser.add_argument("--out", help=".npy filepath where triplets should be stored")
    parser.add_argument("--seed", help="random seed for consistency", default=1234)
    args = parser.parse_args()
    
    main(args)