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

### TODO
- [x] Release training code.
- [x] Release data processing scripts.
- [ ] Release evaluation scripts.
- [ ] Release the pretrained models.
- [ ] Testing.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>
    @article{vora2025hierarchical,
        title={Hierarchical Transformers for Unsupervised 3D Shape Abstraction},
        author={Vora, Aditya and Goli, Lily and Tagliasacchi, Andrea and Zhang, Hao},
        journal={arXiv preprint arXiv:2510.27088},
        year={2025}
    }
</code></pre>
  </div>
</section>