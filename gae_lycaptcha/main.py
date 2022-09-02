import flask, logging, os, io, random, base64, json

import google.cloud.logging
google.cloud.logging.Client().setup_logging()

# get cloud debugger
try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=False
  )
except ImportError:
  print('cannot load cloud debugger')

app = flask.Flask(__name__)

###### START: ml parts
import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import io, random, base64, json

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

def get_one_hot_labels(labels, n_classes):
    import torch.nn.functional as F
    return F.one_hot(labels, num_classes=n_classes)

def combine_vectors(x, y):
    combined = torch.cat((x.float(), y.float()), 1)
    return combined

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan

mnist_shape = (1, 28, 28)
n_classes = 10
n_epochs = 200
z_dim = 64
device = 'cpu'

generator_input_dim, _ = get_input_dimensions(z_dim, mnist_shape, n_classes)

model = Generator(input_dim=generator_input_dim).to(device)
model.load_state_dict(torch.load('./state_dict', map_location=torch.device(device)))
model.eval()

def generate_image(numbers):
  noises_list = [ get_noise(1, z_dim) for _ in numbers ]
  noises = torch.Tensor(len(numbers), z_dim)
  torch.cat(noises_list, out=noises)

  labels_list = [ get_one_hot_labels(torch.Tensor([number]).long(), n_classes) for number in numbers ]
  labels = torch.Tensor(len(numbers), n_classes)
  torch.cat(labels_list, out=labels)

  input = combine_vectors(noises.to(device), labels.to(device))
  output = model(input)
  show_tensor_images(output, num_images=len(numbers), nrow=4, show=False)
### END ml parts

@app.route('/')
def generate_pic():
    input = [ random.randrange(10) for _ in range(4) ]
    generate_image(input)
    plt.axis('off')

    rt = io.BytesIO()
    plt.savefig(rt, format='jpg')
    rt.seek(0)
    b64jpg = base64.b64encode(rt.read())

    print(json.dumps({
        'input': input,
        'jpg': b64jpg.decode()
    }))
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)