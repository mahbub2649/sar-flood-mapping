# Results: unet_r34

Water IoU (95% bootstrap interval over chips), F1, precision and recall.

| Split | Method | IoU | 95% CI | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| test | U-Net (this repo) | 0.676 | 0.560-0.763 | 0.807 | 0.821 | 0.793 |
| test | Fixed VH threshold (-23.5 dB) | 0.534 | 0.422-0.616 | 0.696 | 0.742 | 0.656 |
| test | Otsu per image | 0.231 | 0.164-0.302 | 0.376 | 0.253 | 0.729 |
| bolivia | U-Net (this repo) | 0.590 | 0.405-0.689 | 0.742 | 0.840 | 0.665 |
| bolivia | Fixed VH threshold (-23.5 dB) | 0.543 | 0.389-0.635 | 0.704 | 0.868 | 0.592 |
| bolivia | Otsu per image | 0.404 | 0.189-0.603 | 0.576 | 0.432 | 0.863 |

## IoU by region (test split)

| Region | U-Net | Fixed threshold |
|---|---|---|
| Ghana | 0.556 | 0.455 |
| India | 0.675 | 0.598 |
| Mekong | 0.847 | 0.726 |
| Nigeria | 0.865 | 0.715 |
| Pakistan | 0.314 | 0.151 |
| Paraguay | 0.641 | 0.465 |
| Somalia | 0.505 | 0.258 |
| Spain | 0.713 | 0.641 |
| Sri-Lanka | 0.832 | 0.724 |
| USA | 0.559 | 0.373 |
