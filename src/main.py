from domains.sphere import Sphere
import torch

center = torch.tensor(0.)
radius = torch.tensor(2.)
sphere = Sphere(3, center, radius)