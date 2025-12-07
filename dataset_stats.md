# Dataset Stats

### Stats on our 40% Split of the main AI-GenBench Dataset

## Split Counts

| Split | Count   |
|-------|--------:|
| Training | 115,200 |
| Validation | 28,340 |
|────────────|
| Total | 143,540 |

## Image Type Counts

| Type | Count   |
|-------|--------:|
| Real | 71,770 |
| Fake | 71,770 |
|──────|
| Total | 143,540 |

## Image Type by Split Counts

| Split/Type | Count   |
|-------|--------:|
| Real Train | 57,600 |
| Fake Train | 57,600 |
| Real Validation | 14,170 |
| Fake Validation | 14,170 |
|─────────────────|
| Total | 143,540 |

## Real Image Dataset Sources

### Total Real from Souces

| Real Source | Count   |
|-------|--------:|
| Laion | 41,043 |
| Coco | 30,727 |
|──────|
| Total | 71,770 |

### Real Split Sources

| Real Split Source | Count   |
|-------|--------:|
| Train Laion | 28,825 |
| Train Coco | 28,775 |
| Validation Laion | 12,218 |
| Validation Coco | 1,952 |
|─────────────────|
| Total | 71,770 |

* note the low number of coco validation images. This is due to the original coco validation dataset only having 5,000 images

## Fake Image Dataset Sources

The Fake Images are sourced from 12 datasets, and 36 total generators. Some of the datasets have different splits listed, and some of the generators have different releases listed. 

### Fake Generator Counts

| Generator                     | Count   |
|-------------------------------|--------:|
| ADM                           |   1,976 |
| BigGAN                        |   1,976 |
| CIPS                          |   2,018 |
| Cascaded Refinement Networks  |   2.008 |
| CycleGAN                      |   1,953 |
| DALL-E 3                      |   2,086 |
| DDPM                          |   2,033 |
| DeepFloyd IF                  |   2,008 |
| Denoising Diffusion GAN       |   2,033 |
| Diffusion GAN (ProjectedGAN)  |   2,024 |
| Diffusion GAN (StyleGAN2)     |   1,942 |
| FLUX 1 Dev                    |   1,973 |
| FLUX 1 Schnell                |   2,026 |
| FaceSynthetics                |   1,960 |
| GANformer                     |   1,935 |
| GauGAN                        |   2,005 |
| Glide                         |   1,927 |
| IMLE                          |   1,978 |
| LaMa                          |   1,999 |
| Latent Diffusion              |   2,025 |
| MAT                           |   2,024 |
| Midjourney                    |   1,939 |
| Palette                       |   1,912 |
| ProGAN                        |   1,950 |
| ProjectedGAN                  |   1,997 |
| SN-PatchGAN                   |   2,043 |
| Stable Diffusion 1.4          |   1,994 |
| Stable Diffusion 1.5          |   1,982 |
| Stable Diffusion 2.1          |   1,990 |
| Stable Diffusion XL 1.0       |   1,961 |
| StarGAN                       |   1,998 |
| StyleGAN1                     |   2,062 |
| StyleGAN2                     |   1,969 |
| StyleGAN3                     |   2,025 |
| VQ-Diffusion                  |   2,025 |
| VQGAN                         |   2,019 |
|───────────────────────────────|
| Total | 71,770 |

### Fake Image Origin Dataset Counts

| Origin Dataset             | Count   |
|---------------------------|--------:|
| Aeroblade                 |     399 |
| Artifact                  |  31,881 |
| DDMD                      |   6,343 |
| DMimageDetection/test     |   2,706 |
| DMimageDetection/train    |     187 |
| DMimageDetection/valid    |     208 |
| DRCT                      |   2,210 |
| ELSA_D3/train             |   2,269 |
| ELSA_D3/valid             |   2,289 |
| Forensynths/test          |   8,900 |
| Forensynths/train         |     113 |
| Forensynths/valid         |     113 |
| GenImage/train            |   5,481 |
| GenImage/val              |     210 |
| Imageinet                 |   2,464 |
| Polardiffshield           |     785 |
| SFHQ-T2I                  |   4,403 |
| Synthbuster               |     809 |
|───────────────────────────|
| Total | 71,770 |

## Fake Training Split Generators and Sources

| Generator                    | Count |
| ---------------------------- | ----: |
| DALL-E 3                     |  1680 |
| CIPS                         |  1664 |
| SN-PatchGAN                  |  1644 |
| StyleGAN1                    |  1637 |
| Latent Diffusion             |  1635 |
| MAT                          |  1634 |
| Diffusion GAN (ProjectedGAN) |  1630 |
| FLUX 1 Schnell               |  1628 |
| StyleGAN3                    |  1623 |
| DeepFloyd IF                 |  1621 |
| DDPM                         |  1619 |
| VQ-Diffusion                 |  1618 |
| StarGAN                      |  1616 |
| GauGAN                       |  1612 |
| Cascaded Refinement Networks |  1609 |
| StyleGAN2                    |  1606 |
| ProjectedGAN                 |  1603 |
| ADM                          |  1602 |
| Stable Diffusion 2.1         |  1600 |
| Denoising Diffusion GAN      |  1597 |
| VQGAN                        |  1594 |
| BigGAN                       |  1590 |
| Stable Diffusion 1.4         |  1587 |
| FLUX 1 Dev                   |  1586 |
| LaMa                         |  1584 |
| IMLE                         |  1584 |
| Stable Diffusion 1.5         |  1579 |
| CycleGAN                     |  1575 |
| GANformer                    |  1566 |
| Stable Diffusion XL 1.0      |  1565 |
| FaceSynthetics               |  1564 |
| ProGAN                       |  1564 |
| Glide                        |  1561 |
| Diffusion GAN (StyleGAN2)    |  1561 |
| Midjourney                   |  1545 |
| Palette                      |  1517 |
|──────────────────────────────|
| Total | 57,600 |

| Origin Dataset         | Count |
| ---------------------- | ----: |
| Artifact               | 25577 |
| Forensynths/test       |  7160 |
| DDMD                   |  5073 |
| GenImage/train         |  4400 |
| SFHQ-T2I               |  3530 |
| DMimageDetection/test  |  2205 |
| Imaginet               |  1968 |
| ELSA_D3/valid          |  1845 |
| ELSA_D3/train          |  1822 |
| DRCT                   |  1765 |
| Synthbuster            |   645 |
| Polardiffshield        |   629 |
| Aeroblade              |   315 |
| DMimageDetection/valid |   171 |
| GenImage/val           |   167 |
| DMimageDetection/train |   144 |
| Forensynths/train      |    96 |
| Forensynths/valid      |    88 |
|────────────────────────|
| Total | 57,600 |

## Fake Validation Split Generators and Sources

| Generator                    | Count |
| ---------------------------- | ----: |
| Denoising Diffusion GAN      |   436 |
| StyleGAN1                    |   425 |
| VQGAN                        |   425 |
| LaMa                         |   415 |
| DDPM                         |   414 |
| Stable Diffusion 1.4         |   407 |
| DALL-E 3                     |   406 |
| Stable Diffusion 1.5         |   403 |
| VQ-Diffusion                 |   402 |
| StyleGAN3                    |   402 |
| Cascaded Refinement Networks |   399 |
| SN-PatchGAN                  |   399 |
| FLUX 1 Schnell               |   398 |
| Stable Diffusion XL 1.0      |   396 |
| FaceSynthetics               |   396 |
| Palette                      |   395 |
| Midjourney                   |   394 |
| IMLE                         |   394 |
| Diffusion GAN (ProjectedGAN) |   394 |
| ProjectedGAN                 |   394 |
| GauGAN                       |   393 |
| Stable Diffusion 2.1         |   390 |
| MAT                          |   390 |
| Latent Diffusion             |   390 |
| DeepFloyd IF                 |   387 |
| FLUX 1 Dev                   |   387 |
| BigGAN                       |   386 |
| ProGAN                       |   386 |
| StarGAN                      |   382 |
| Diffusion GAN (StyleGAN2)    |   381 |
| CycleGAN                     |   378 |
| ADM                          |   374 |
| GANformer                    |   369 |
| Glide                        |   366 |
| StyleGAN2                    |   363 |
| CIPS                         |   354 |
|──────────────────────────────|
| Total | 14,170 |

| Origin Dataset         | Count |
| ---------------------- | ----: |
| Artifact               |  6304 |
| Forensynths/test       |  1740 |
| DDMD                   |  1270 |
| GenImage/train         |  1081 |
| SFHQ-T2I               |   873 |
| DMimageDetection/test  |   501 |
| Imaginet               |   496 |
| ELSA_D3/train          |   447 |
| DRCT                   |   445 |
| ELSA_D3/valid          |   444 |
| Synthbuster            |   164 |
| Polardiffshield        |   156 |
| Aeroblade              |    84 |
| GenImage/val           |    43 |
| DMimageDetection/train |    43 |
| DMimageDetection/valid |    37 |
| Forensynths/valid      |    25 |
| Forensynths/train      |    17 |
|────────────────────────|
| Total | 14,170 |
