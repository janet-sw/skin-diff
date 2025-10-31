This is the code implementation for the paper [**"From Majority to Minority: A Diffusion-based Augmentation for Underrepresented Groups in Skin Lesion Analysis"**](https://workshop.isic-archive.com/2024/paper_wang.pdf), presented at the @[MICCAI 2024 Workshops - ISIC Skin Image Analysis Workshop](https://workshop.isic-archive.com/2024/). Our work received the **Honorable Mention Award**!



### Dependencies

We use Hugging Face's `diffusers` library to train a Stable Diffusion model via **Textual Inversion** and **LoRA**. Before running the script, execute the following lines to install necessary dependencies:

```bash
conda create -n skin-diff python=3.12
conda activate skin-diff
pip install -r requirements.txt
```



### Concept Discovery via Textual Inversion

To find suitable textual embedding for each condition using Textual Inversion, run the first half of the notebook `/notebooks/finetune.ipynb`.

You can verify if the training is successful by generating images with the new tokens via text-to-image. For more details, see the inference section at https://huggingface.co/docs/diffusers/training/text_inversion.



### Fine-grained Detail Enhancement with LoRA

To fine-tune a Stable Diffusion model with the new tokens using LoRA, run the second half of the notebook `/notebooks/finetune.ipynb`. Here, we use a json file to manage the training data for LoRA.

For more details, see https://huggingface.co/docs/diffusers/training/lora



### Image Generation 

To generate images, please run the notebook `/notebooks/inference.ipynb`. In our experiment, we used the `StableDiffusionImg2ImgPipeline` from `diffusers`. Key parameters to tune include: 

- `strength`
- `guidance_scale`
- `num_inference_steps`
- `num_images_per_prompt`

Detailed explanations for these parameters are available in the [Hugging Face API documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/img2img).



### Classifier Training

We adapted the classifier training code from [this repository](https://github.com/mattgroh/fitzpatrick17k). To train the classifier, run

```bash
python train_classifier.py 30 full
```



Here, `30` refers to the number of epochs, and `full` specifies the full dataset. Modify the script as needed; areas requiring changes are marked with the keyword "change."



### Cite

If you find this code implementation helpful, please consider citing our work

```latex
@article{wang2024majority,
  title={From Majority to Minority: A Diffusion-based Augmentation for Underrepresented       Groups in Skin Lesion Analysis},
  author={Wang, Janet and Chung, Yunsung and Ding, Zhengming and Hamm, Jihun},
  journal={arXiv preprint arXiv:2406.18375},
  year={2024}
}
```