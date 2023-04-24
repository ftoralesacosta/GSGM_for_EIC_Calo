# Fast Point Cloud Diffusion (FPCD)

This is a fork of the official implementation of the FPCD paper that uses a diffusion model to generate particle jets while [progressive distillation](https://arxiv.org/abs/2202.00512) is used to accelerate the generation. This fork is intended to be used to generate point-cloud calorimeter data. The calorimeter is designed for the upcoming Electron Ion Collider, and more information for this calorimeter can be found [here](https://github.com/eiccodesign/generate_data).

![Visualization of FPCD](./assets/FPCD.png)

# Training a new model

To train a new model from scratch, first, obtain data from the [eic/generate_data](https://github.com/eiccodesign/generate_data) repository. Then convert the root file(s) to HDF5 with [this converter](https://github.com/eiccodesign/generate_data/blob/main/to_hdf5/h5_for_FPCD_converter.cc)
The baseline model can be trained with:
```bash
cd scripts
python train.py [--big]
```
with optiional --big flag to choose between the 30 or 150 particles dataset.
After training the baseline model, you can train the distilled models with:
```bash
python train.py --distill --factor 2
```
This step will train a model that decreases the overall number of time steps by a factor 2. Similarly, you can load the distilled model as the next teacher and run the training using ```--factor 4``` and so on to halve the number of evaluation steps during generation.

To reproduce the plots provided in the paper, you can run:
```bash
python plot_jet.py [--distill --factor 2] --sample
```
The command will generate new observations with optional flags to load the distilled models. Similarly, if you already have the samples generated and stored, you can omit the ```--sample``` flag to skip the generation.

# Plotting and Metrics

The calculation os the physics inspired metrics is taken directly from the [JetNet](https://github.com/jet-net/JetNet) repository, thus also need to be cloned. Notice that while our implementation is carried out using TensorFlow while the physics inspired metrics are implemented in Pytorch.

Out distillation model is partially based on a [Pytorch implementation](https://github.com/Hramchenko/diffusion_distiller).

