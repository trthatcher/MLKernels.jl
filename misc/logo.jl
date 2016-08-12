using Plots
gr() #pyplot() #plotlyjs()

using MLKernels

xmin, xmax = (0.25, 1.5)
ymin, ymax = (0.25, 1.5)

κ = PeriodicKernel() + PolynomialKernel(1.0, 0.0, 1) 
x = xmin:0.005:xmax
y = ymin:0.005:ymax
k(x,y) = kernel(κ, x, y)

p = heatmap(x, y, k, fill=true, colorbar=:none)
plot(p)
