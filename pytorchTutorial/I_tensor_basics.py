import torch
import numpy as np

x = torch.empty(1)
print(x)
x = torch.empty(3)
print(x)
x = torch.empty(2, 3)
print(x)
x = torch.empty(2, 2, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3)
print(x)

print(x.size())

print(x.dtype)

x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

print(x.dtype)

x = torch.tensor([5.5, 3])
print(x.size())

x = torch.tensor([5.5, 3], requires_grad=True)

y = torch.rand(2, 2)
x = torch.rand(2, 2)

# elementwise addition
z = x + y
# torch.add(x,y)
# y.add_(x) modifies variable y

# subtraction
z = x - y
z = torch.sub(x, y)

# multiplication
z = x * y
z = torch.mul(x, y)

# division
z = x / y
z = torch.div(x, y)

# Slicing
x = torch.rand(5, 3)
print(x)
print(x[:, 0])  # all rows, column 0
print(x[1, :])  # row 1, all columns
print(x[1, 1])  # element at 1, 1

# Get the actual value if only 1 element in your tensor
print(x[1, 1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())

# Numpy
a = torch.ones(5)
print(a)

# torch to numpy with .numpy()
b = a.numpy()
print(b)
print(type(b))

# If the Tensor is on the CPU (not the GPU),
# both objects will share the same memory location, so changing one
# will also change the other
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

a += 1
print(a)
print(b)

# by default all tensors are created on the CPU,
# but you can also move them to the GPU (only if it's available )
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    # z = z.numpy() # not possible because numpy cannot handle GPU tenors
    # move to CPU again
    z.to("cpu")  # ``.to`` can also change dtype together!
    # z = z.numpy()
