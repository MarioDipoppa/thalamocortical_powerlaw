import jax
import jax.numpy as jnp
from jax import lax

# Show devices
print("Devices:", jax.devices())

# Define function to sum across devices
def f(x):
    return lax.psum(x, "i")

# PMAP the function across devices, axis_name must match lax.psum
f_pmapped = jax.pmap(f, axis_name="i")

# Input shape: (num_devices, features)
# Each row will go to a separate GPU
num_devices = jax.device_count()
x = jnp.ones((num_devices, 4))

# Run pmapped function
result = f_pmapped(x)
print(result)
