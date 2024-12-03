# %%
import os
import tempfile

import einops
import imageio
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# get mnist dataset


device = (
    t.device("mps")
    if t.backends.mps.is_available()
    else t.device("cuda")
    if t.cuda.is_available()
    else t.device("cpu")
)


class VAE(nn.Module):
    def __init__(self, image_features=784, latent_size=10):
        super().__init__()
        self.latent_size = latent_size
        steps = [2560, 320]
        self.encoder = nn.Sequential(
            nn.Linear(in_features=image_features, out_features=steps[0]),
            nn.ReLU(),
            nn.Linear(in_features=steps[0], out_features=steps[1]),
            nn.ReLU(),
            nn.Linear(in_features=steps[1], out_features=latent_size),
        )
        self.mean_from_encoded = nn.Linear(
            in_features=latent_size, out_features=latent_size
        )
        self.cov_diag_from_encoded = nn.Linear(
            in_features=latent_size, out_features=latent_size
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=steps[1]),
            nn.ReLU(),
            nn.Linear(in_features=steps[1], out_features=steps[0]),
            nn.ReLU(),
            nn.Linear(in_features=steps[0], out_features=image_features),
            nn.Sigmoid(),
        )

    def forward(self, images: t.Tensor):
        images = einops.rearrange(
            images, "batch chan width height -> batch (chan width height)"
        )
        z, *_ = self.encode(images)
        decoded = self.decoder(z)
        return einops.rearrange(
            decoded, "batch (width height) -> batch width height", width=28, height=28
        )

    def encode(self, images: t.Tensor):
        batch_size, image_size = images.shape
        latent = self.encoder(images)
        assert latent.shape[-1] == self.latent_size
        mean_from_encoded = self.mean_from_encoded(latent)
        next(self.parameters()).device
        zeta = t.randn((batch_size, self.latent_size)).to(device)
        cov_diag_from_encoded = self.cov_diag_from_encoded(latent)
        z = mean_from_encoded + cov_diag_from_encoded * zeta
        z.relu_()
        return z, zeta, mean_from_encoded, cov_diag_from_encoded

    @t.inference_mode()
    @staticmethod
    def random_different_labels(labels: t.Tensor, num_classes: int = 10):
        """
        Returns new labels, ensuring that none are the same as the original labels.
        """
        new_labels = t.randint(0, num_classes, labels.size()).to(device)

        # Generate mask where new_labels are the same as the original labels
        mask = new_labels == labels

        # Re-generate new labels only where the mask is True
        while mask.any():
            new_labels[mask] = t.randint(0, num_classes, (mask.sum(),)).to(device)  # type: ignore
            mask = new_labels == labels

        return new_labels

    def encode_and_mask(self, images: t.Tensor, labels: t.Tensor):
        encoded_unmasked, zeta, mean_from_encoded, cov_diag_from_encoded = self.encode(
            images
        )
        mask_one_hot = F.one_hot(labels, num_classes=self.latent_size).float()  # type: ignore
        # mask_one_hot += (
        #    0.2
        #    * F.one_hot(
        #        VAE.random_different_labels(labels), num_classes=self.latent_size
        #    ).float()
        # )  # type: ignore
        unmask_one_hot = 1 - mask_one_hot
        encoded = (
            mask_one_hot * encoded_unmasked + unmask_one_hot * encoded_unmasked.detach()
        )
        return encoded, zeta, mean_from_encoded, cov_diag_from_encoded

    def calculate_loss(self, images: t.Tensor, labels=None):
        if labels is not None:
            encoded, zeta, mean_from_encoded, cov_diag_from_encoded = (
                self.encode_and_mask(images, labels)
            )
        else:
            encoded, zeta, mean_from_encoded, cov_diag_from_encoded = self.encode(
                images
            )

        decoded = self.decoder(encoded)
        # mse loss plus kl div loss to push the latent space to be in the distribution
        mse_loss = ((images - decoded).norm(dim=-1) ** 2).mean()
        kl_div_loss = (
            mean_from_encoded**2 + cov_diag_from_encoded.exp() - cov_diag_from_encoded
        ).mean()
        loss = 0.3 * mse_loss + kl_div_loss
        return loss, mse_loss, kl_div_loss


class EncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        x = x.view(x.size(0), -1)
        latent = self.vae.encoder(x)
        mean = self.vae.mean_from_encoded(latent)
        log_var = self.vae.cov_diag_from_encoded(latent)
        return mean, log_var


class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        decoded = self.vae.decoder(z)
        return decoded.view(-1, 1, 28, 28)


mnist_data = datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)

# put everything on gpu first
new_mnist_data = []
for image, labels in mnist_data:
    new_mnist_data.append((image.to(device), labels))
dataloader = DataLoader(new_mnist_data, batch_size=128, shuffle=True)  # type: ignore
validation_data = datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor()
)
validation_dataloader = DataLoader(validation_data, batch_size=128, shuffle=True)  # type: ignore
vae = VAE()
vae.to(device)
optim = t.optim.Adam(vae.parameters(), lr=1e-3)
# load everything onto the gpu first

for epoch in range(20):
    for images, labels in (pbar := tqdm.tqdm(dataloader)):
        images = einops.rearrange(
            images, "batch chan width height -> batch (chan width height)"
        )
        images = images.to(device)
        labels = labels.to(device)
        loss, mse_loss, kl_div_loss = vae.calculate_loss(images, labels)
        loss.backward()
        pbar.set_postfix({"mse loss": mse_loss.item()})
        optim.step()
        optim.zero_grad()
    if t.backends.mps.is_available():
        t.mps.empty_cache()
# %%


print("ENCODING SOME IMAGES")
image_0 = next(iter(validation_dataloader))[0].to(device)
image_0_vae = vae(image_0)
for i in range(20):
    plt.imshow(image_0[i].squeeze().cpu().numpy())
    plt.show()
    plt.imshow(image_0_vae[i].squeeze().detach().cpu().numpy())
    plt.show()

# explore the latent space
# print("EXPLORING THE LATENT SPACE")
# n = 10
# random_latents = t.randn((n, vae.latent_size)).to(device)
# random_latents_decoded = vae.decoder(random_latents).reshape((n, 28, 28)).detach().cpu()
# for i in range(n):
#    plt.imshow(random_latents_decoded[i])
#    plt.show()

# print("Generating 0-9 from the interpretable dimensions")
# for i in range(10):
#    v = 1 * F.one_hot(t.tensor(i), num_classes=10).float().to(device)
#    print(f"inputting {v} as latent space and decoding VAE:")
#    v_decoded = vae.decoder(v).reshape((1, 28, 28)).detach().cpu()
#    plt.imshow(v_decoded[0])
#    plt.show()


def decode_and_show(vec, scale=1.0, show=True):
    if show:
        print("decoding", vec)

    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(  # type: ignore
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1, 3]}
    )
    ax1.axis("off")
    ax2.axis("off")

    # Convert the input tensor to a numpy array for the heatmap
    vector = vec.detach().cpu().numpy()

    # Plot the heatmap
    ax1.imshow(vector.reshape(1, -1), aspect="auto")
    ax1.set_title("Vector to decode")

    # Decode and show the image
    v = scale * t.tensor(vec).float().to(device)
    v_decoded = vae.decoder(v).reshape((1, 28, 28)).detach().cpu()
    ax2.imshow(v_decoded[0])
    ax2.set_title("Decoded image")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


for i in range(10):
    start = t.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    start[i] = 1.0
    decode_and_show(start, show=True)


# %%
def create_gif_from_figures(figures, output_filename="animation.gif", fps=60):
    """
    Create a GIF from a list of Matplotlib figures.
    Args:
    figures (list): List of Matplotlib figure objects
    output_filename (str): Name of the output GIF file
    fps (int): Frames per second for the output GIF
    Returns:
    str: The filename of the created GIF
    """
    # Create a temporary directory to store image files
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save figures as images
        image_files = []
        for i, fig in enumerate(figures):
            filename = os.path.join(tmpdirname, f"figure_{i}.png")
            fig.savefig(filename)
            image_files.append(filename)
            plt.close(fig)  # Close the figure to free up memory

        # Calculate duration based on fps
        duration = 1 / fps

        # Create GIF
        with imageio.get_writer(output_filename, mode="I", duration=duration) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)  # type: ignore

    return output_filename


def lerp(start, end, pct):
    """
    pct should be between 0 and 1
    """
    return start + (end - start) * pct


figs = []
for i in range(9):
    start = t.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    start[i] = 1.5
    end = t.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    end[i + 1] = 1.5
    for pct in t.arange(0, 1, 0.025):
        figs.append(decode_and_show(lerp(start, end, pct), show=False))
print(len(figs))
create_gif_from_figures(figs, fps=300)

# %%
# save the weights
SAVE = True
if SAVE:
    t.save(vae.state_dict(), "vae.pth")
else:
    vae = VAE()
    vae.to(device)
    vae.load_state_dict(t.load("vae.pth", map_location=device))
vae.eval()
encoder_only = EncoderWrapper(vae)
decoder_only = DecoderWrapper(vae)
t.onnx.export(
    vae,
    t.randn(1, 1, 28, 28).to(device),
    "vae_full.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

# Export the encoder
t.onnx.export(
    encoder_only,
    t.randn(1, 1, 28, 28).to(device),
    "vae_encoder.onnx",
    input_names=["input"],
    output_names=["mean", "log_var"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "mean": {0: "batch_size"},
        "log_var": {0: "batch_size"},
    },
    opset_version=11,
)

# Export the decoder
t.onnx.export(
    decoder_only,
    t.randn(1, 10).to(device),
    "vae_decoder.onnx",
    input_names=["z"],
    output_names=["output"],
    dynamic_axes={"z": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print("ONNX models exported successfully.")
