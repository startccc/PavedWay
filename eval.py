from unet import UNet


model = UNet(n_class=1)
model = model.to(device)

