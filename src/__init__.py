"""Verejne API balicku ``src`` pro cviceni 10.

Obsahuje:

* ``Activation`` (ABC) -- spolecne rozhrani aktivacnich funkci, NOVE s metodou
  ``derivative`` (brana z Cviceni 09 + rozsireni).
* ``Step`` / ``Sigmoid`` / ``Tanh`` / ``ReLU`` / ``ReLU6`` -- konkretni
  aktivace (brana z Cviceni 09 + rozsireni o ``derivative``).
* ``make_activation`` -- tovarni funkce: jmeno aktivace -> instance
  ``Activation`` (brana z Cviceni 09).
* ``Linear`` -- afinni vrstva ``x @ weights + bias``, NOVE s metodou ``update``
  (brana z Cviceni 09 + rozsireni).
* ``Neuron`` -- slozeni ``Linear`` a injektovane ``Activation``; slouzi jako
  vrstva site (brana z Cviceni 09, beze zmeny).
* ``Sequential`` -- retezeni vrstev, NOVE se zaznamem ``z_`` a metodami
  ``backward`` a ``update`` (brana z Cviceni 09 + rozsireni).
* ``Loss`` (ABC) / ``MSE`` / ``BCE`` / ``make_loss`` -- ztratove funkce (NOVE).
* ``WeightsInitializer`` (ABC) / ``RandomUniformInit`` / ``XavierInit`` /
  ``HeInit`` / ``PyTorchDefaultInit`` / ``make_weights_initializer`` --
  inicializace vah (NOVE).
* ``Trainer`` -- trenovaci wrapper s logovanim (NOVE, predvyplneno).

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.activations import Activation, ReLU, ReLU6, Sigmoid, Step, Tanh, make_activation
from src.initializers import (
    HeInit,
    PyTorchDefaultInit,
    RandomUniformInit,
    WeightsInitializer,
    XavierInit,
    make_weights_initializer,
)
from src.linear import Linear
from src.losses import BCE, MSE, Loss, make_loss
from src.network import Sequential
from src.neuron import Neuron
from src.trainer import Trainer

__all__ = [
    "Activation",
    "Step",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "ReLU6",
    "make_activation",
    "Linear",
    "Neuron",
    "Sequential",
    "Loss",
    "MSE",
    "BCE",
    "make_loss",
    "WeightsInitializer",
    "RandomUniformInit",
    "XavierInit",
    "HeInit",
    "PyTorchDefaultInit",
    "make_weights_initializer",
    "Trainer",
]
