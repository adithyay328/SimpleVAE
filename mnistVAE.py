"""
A simple implementation of VAEs
for MNIST. With this, you should
be able to encode a digit, sample
a new digit from it, and get something
slightly different from it.

Later on, LDMs can build off of this
to do diffusion in the latent space,
which'll be better conditioned due to
the smoothness benefits of training
a VAE
"""
import argparse

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pydantic

class VAEConfig(pydantic.BaseModel):
  randomSeed : int = 42
  latentDims : int = 50
  lr : float = 1e-5
  bs : int = 256
  numEpochs : int = 30
  numHiddenLayers : int = 2
  hiddenDim : int = 256
  droputFrac : float = 0.3
  l2Decay : float = 0.0
  blockNormType : str = "layer"
  klWeight : float = 0.1

# Parse in the config file; this
# contains most of the variations
# in the model
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
args = parser.parse_args()

if args.config is not None:
  with open(args.config, "r") as configFile:
    config = VAEConfig.model_validate_json(configFile.read())

else:
  config = VAEConfig()

print(config)

# Pull out config
SEED = config.randomSeed
LATENT_DIMS = config.latentDims
LR = config.lr
BS = config.bs
NUM_EPOCHS = config.numEpochs
NUM_HIDDEN_LAYERS = config.numHiddenLayers
HIDDEN_DIM = config.hiddenDim
DROP_FRAC = config.droputFrac
L2_DECAY = config.l2Decay
BLOCK_NORM_TYPE = config.blockNormType
KL_WEIGHT = config.klWeight

# The hidden activation really doesn't matter,
# but I'm gonna just use Mish here. GELU's good,
# so is RELU
ACT = nn.GELU

np.random.seed(SEED)
torch.manual_seed(SEED)

mnistTrain = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
mnistTest = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

device = "cuda" if torch.cuda.is_available() else "cpu"

# Reduce f32 precision
torch.set_float32_matmul_precision('high')

"""
For MNIST, we need to estimate
a mean and co-variance. To make
the co-variance much easier to compute
we assume that all dimensions are un-correlated,
leading to a diagonal covar matrix that is the
same shape as the latent space.

So, after all that, we'll be predicting:
1. mean
2. diagonal terms of the co-variance(the
variance per dimension). To make parametrization
more stable, we'll predict the logarithm
of the variance.

I'll make both the encoder and decoder an MLP,
just so there's less tweaking needed on my end
than something fancy like a U-Net.
"""
class MLPBlock(nn.Module):
  """
  A simple residual hidden
  block in an MLP.
  """
  def __init__(self, inDims : int, outDims : int = None, dropoutFrac : float = 0.0, layerNormType : str = "layer"):
    super().__init__()

    outDims = outDims if outDims is not None else inDims

    self.seq = nn.Sequential(
      nn.Linear(inDims, outDims),
      ACT(),
      nn.Dropout(dropoutFrac)
    )

    if layerNormType == "layer":
      self.norm = nn.LayerNorm(outDims)

  def forward(self, X):
    prevOut = self.seq(X) + X
    if "norm" in self.__dict__:
      return self.norm(prevOut)
    else:
      return prevOut

# Uncomment to disable compilation of models;
# makes debug messages easier to read.
# torch.compile = lambda x: x

encoder = torch.compile(nn.Sequential(
  nn.Flatten(),
  nn.Linear(28 * 28, HIDDEN_DIM),
  ACT(),
  nn.Dropout(DROP_FRAC),
  *[MLPBlock(HIDDEN_DIM, dropoutFrac = DROP_FRAC, layerNormType = BLOCK_NORM_TYPE) for _ in range(NUM_HIDDEN_LAYERS)],
  # The output is the mean
  # and log variance concatted,
  # just break them after
  nn.Linear(HIDDEN_DIM, LATENT_DIMS * 2)
)).to(device)

decoder = torch.compile(nn.Sequential(
  nn.Linear(LATENT_DIMS, HIDDEN_DIM),
  ACT(),
  nn.Dropout(DROP_FRAC),
  *[MLPBlock(HIDDEN_DIM, dropoutFrac = DROP_FRAC, layerNormType = BLOCK_NORM_TYPE) for _ in range(NUM_HIDDEN_LAYERS)],
  nn.Linear(HIDDEN_DIM, 28 * 28),
  nn.Sigmoid()
)).to(device)

def getMeanVarsFromInput(X, encoderModule):
  """
  Given an input image/batch and
  an encoder module, returns
  a tuple of (mean, variance)
  for that sample.

  To get variance, we just take exp
  of the model's predicted variance,
  which makes it so that it's treated
  as log space.
  """
  Y = encoderModule(X)
  return Y[:, :LATENT_DIMS], torch.exp(Y[:, LATENT_DIMS:])

def sampleFromMeanVar(mean, var, decoder):
  """
  Given a mean and a variance, generates
  a sample in the latent space
  and decodes it to get the output.
  """
  # First, generate a random tensor,
  # and scale it with variance by mulling
  # with stanard deviation
  scaledNoise = torch.sqrt(var) * torch.randn_like(var)

  # Add to mean, and then pass into
  # decoder
  return decoder(mean + scaledNoise)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR, weight_decay=L2_DECAY)

def train():
  crossEpochTrainLosses = []
  crossEpochTestLosses = []

  for epochIdx in range(NUM_EPOCHS):
    currEpochTrainLosses = []
    currEpochTestLosses = []

    # Set model to train mode
    encoder.train()
    decoder.train()

    for batchIdx, batch in enumerate(DataLoader(mnistTrain, batch_size=BS, shuffle=True)):
      X, _ = batch
      X = X.to(device)
      # Get the mean and variance
      mean, var = getMeanVarsFromInput(X, encoder)
      # Sample from the mean and variance
      XSampled = sampleFromMeanVar(mean, var, decoder)
  
      # Compute the loss

      # The loss is the reconstruction loss
      # plus the KL divergence between the
      # prior and the posterior

      reconstructionLoss = nn.functional.mse_loss(XSampled, X.flatten(start_dim=1))
      klDivergence = -0.5 * torch.mean(1 + torch.log(var) - mean.pow(2) - var)

      loss = reconstructionLoss + KL_WEIGHT * klDivergence
      # Backpropagate
      loss.backward()
      # Optimize
      optimizer.step()

      currEpochTrainLosses.append(loss.item())

    # Now, run testing iterations
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
      for batchIdx, batch in enumerate(DataLoader(mnistTest, batch_size=BS, shuffle=True)):
        X, _ = batch
        X = X.to(device)
        # Get the mean and variance
        mean, var = getMeanVarsFromInput(X, encoder)
        # Sample from the mean and variance
        XSampled = sampleFromMeanVar(mean, var, decoder)

        # Compute the loss
        # The loss is the reconstruction loss
        # plus the KL divergence between the
        # prior and the posterior
        reconstructionLoss = nn.functional.mse_loss(XSampled, X.flatten(start_dim=1))
        klDivergence = -0.5 * torch.mean(1 + torch.log(var) - mean.pow(2) - var)

        loss = reconstructionLoss + KL_WEIGHT * klDivergence
        currEpochTestLosses.append(loss.item())

    # Add means to cross epoch
    crossEpochTrainLosses.append(np.mean(currEpochTrainLosses))
    crossEpochTestLosses.append(np.mean(currEpochTestLosses))

    # Print epoch means
    print(f"Epoch {epochIdx} Train Loss: {crossEpochTrainLosses[-1]} Test Loss: {crossEpochTestLosses[-1]}")

  
  # Plot losses
  plt.plot(crossEpochTrainLosses, label="Train Loss")
  plt.plot(crossEpochTestLosses, label="Test Loss")
  plt.legend()
  plt.savefig("losses.png")

if __name__ == "__main__":
  train()
