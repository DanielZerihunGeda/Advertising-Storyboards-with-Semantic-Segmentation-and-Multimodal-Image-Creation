# Packages
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision.utils import save_image
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix
import argparse
import pdb
import os
import imageio.v2 as iio
import torch.nn.functional as F



def optimize_blend(source_img, mask_img, target_img, output_dir='results/1', ss=300, ts=512, x=200, y=235, gpu_id=0, num_steps=1000, save_video=False):
    os.makedirs(output_dir, exist_ok=True)

    # Make canvas mask
    canvas_mask = make_canvas_mask(x, y, target_img, mask_img)
    canvas_mask = numpy2tensor(canvas_mask, gpu_id)
    canvas_mask = canvas_mask.squeeze(0).repeat(3, 1).view(3, ts, ts).unsqueeze(0)

    # Compute ground-truth gradients
    gt_gradient = compute_gt_gradient(x, y, source_img, target_img, mask_img, gpu_id)

    # Convert numpy images to tensors
    source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(gpu_id)
    target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(gpu_id)
    input_img = torch.randn(target_img.shape).to(gpu_id)

    mask_img = numpy2tensor(mask_img, gpu_id)
    mask_img = mask_img.squeeze(0).repeat(3, 1).view(3, ss, ss).unsqueeze(0)

    # Define optimizer
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    optimizer = get_input_optimizer(input_img)

    # Define loss function
    mse = torch.nn.MSELoss()

    # Save reconstruction process in a video
    if save_video:
        recon_process_video = iio.get_writer(os.path.join(output_dir, 'recon_process.mp4'), format='FFMPEG', mode='I', fps=400)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            # Composite foreground and background
            blend_img = torch.zeros(target_img.shape).to(gpu_id)
            blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)

            # Compute Laplacian gradient of blended image
            pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)

            # Compute gradient loss
            grad_loss = sum(mse(pred_gradient[c], gt_gradient[c]) for c in range(len(pred_gradient)))
            grad_loss /= len(pred_gradient)
            grad_loss *= 1e4

            # Compute style loss
            target_features_style = vgg(mean_shift(target_img))
            target_gram_style = [gram_matrix(y) for y in target_features_style]

            blend_features_style = vgg(mean_shift(input_img))
            blend_gram_style = [gram_matrix(y) for y in blend_features_style]

            style_loss = sum(mse(blend_gram_style[layer], target_gram_style[layer]) for layer in range(len(blend_gram_style)))
            style_loss /= len(blend_gram_style)
            style_loss *= 1e4

            # Compute content loss
            blend_obj = blend_img[:, :, int(x - source_img.shape[2] * 0.5):int(x + source_img.shape[2] * 0.5),
                                    int(y - source_img.shape[3] * 0.5):int(y + source_img.shape[3] * 0.5)]
            source_object_features = vgg(mean_shift(source_img * mask_img))
            blend_object_features = vgg(mean_shift(blend_obj * mask_img))
            content_loss = mse(blend_object_features.relu2_2, source_object_features.relu2_2)

            # Compute TV regularization loss
            tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                      torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
            tv_loss *= 1e-6

            # Total loss and update image
            loss = grad_loss + style_loss + content_loss + tv_loss
            optimizer.zero_grad()
            loss.backward()

            # Write reconstruction process to video
            if save_video:
                foreground = input_img * canvas_mask
                foreground = (foreground - foreground.min()) / (foreground.max() - foreground.min())
                background = target_img * (canvas_mask - 1) * (-1)
                background = background / 255.0
                final_blend_img = foreground + background
                if run[0] < 200:
                    for _ in range(10):
                        recon_process_video.append_data(final_blend_img[0].transpose(0, 2).transpose(0, 1).cpu().data.numpy())
                else:
                    recon_process_video.append_data(final_blend_img[0].transpose(0, 2).transpose(0, 1).cpu().data.numpy())

            # Print loss
            if run[0] % 1 == 0:
                print("run {}:".format(run))
                print('grad: {:4f}, style: {:4f}, content: {:4f}, tv: {:4f}'.format(grad_loss.item(), style_loss.item(), content_loss.item(), tv_loss.item()))
                print()

            run[0] += 1
            return loss

        optimizer.step(closure)

    # Clamp pixel range to 0 ~ 255
    input_img.data.clamp_(0, 255)

    # Make the final blended image
    blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)
    blend_img_np = blend_img.transpose(1, 3).transpose(1, 2).cpu().data.numpy()[0]

    # Save the final blended image
    save_image(blend_img_np.astype(np.uint8), os.path.join(output_dir, 'final_blend.png'))

    # Close the reconstruction process video if saving
    if save_video:
        recon_process_video.close()

    return blend_img_np
