# HiT: Hierarchical Transformers for Unsupervised 3D Shape Abstraction
![](assets/git-teaser.png)

### Aditya Vora, Lily Goli, Andrea Tagliasacchi, Hao Zhang

This repository contains the official implementation of [HiT: Hierarchical Transformers for Unsupervised 3D Shape Abstraction](https://aditya-vora.github.io/HiT/).


### Installation Steps

1. **Clone the Repository:**
   ```sh
   git clone --recursive https://github.com/aditya-vora/HiT
   cd HiT
   ```

2. **Set Up the Conda Environment:**
    ```sh
    conda create -n hit python=3.10;
    conda activate hit
    ```

3. **Install the dependencies inside the conda environment:**
    ```sh
    pip install -r requirements.txt
    ```

### Training

Train HiT on ShapeNet-formatted data (see `preprocess/` for how to build the required `.hdf5` files):

```sh
sh scripts/train_all_cats.sh
```

or invoke `train.py` directly, e.g.:

```sh
python train.py --train --data-dir <path-to-data> --dataset-name shapenet --cats all --exp-name my_experiment
```

### Evaluation

`eval_seg.py` loads a trained checkpoint (`--ckpt`) and evaluates it on the `test` split in one of five modes, selected by passing exactly one of the following flags:

| Flag | Output |
| --- | --- |
| `--recon` | Colored part + full-shape meshes for a single shape (`--shape-id`), via marching cubes. |
| `--mesh` | Colored part + full-shape meshes for every shape in the test set. |
| `--pointcloud` | A colored, per-part-segmented point cloud (`.ply`) for every test shape. |
| `--iou` | Per-shape, per-category, and overall part-segmentation mIoU against ground-truth labels. |
| `--cd` | Per-category and overall Chamfer Distance / F-score against the ground-truth surface. |

```sh
sh scripts/eval_all_cats.sh
```

or directly, e.g. to compute segmentation mIoU:

```sh
python eval_seg.py --iou --ckpt <path-to-checkpoint.pt> --data-dir <path-to-data> --dataset-name shapenet --cats all --splits test
```

Results are written under `<exp-dir>/<exp-name>/<NNN>/`.

### TODO
- [x] Release training code.
- [x] Release data processing scripts.
- [x] Release evaluation scripts.
- [x] Testing.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>
      @inproceedings{vora2026hit,
        title={HiT: Hierarchical Transformers for Unsupervised 3D Shape Abstraction},
        author={Vora, Aditya and Goli, Lily and Tagliasacchi, Andrea and Zhang, Hao},
        booktitle={2026 International Conference on 3D Vision (3DV)},
        pages={1598--1607},
        year={2026},
        organization={IEEE}
      }

</code></pre>
  </div>
</section>