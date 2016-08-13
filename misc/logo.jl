using MLKernels
using PlotlyJS

xmin, xmax = (0.25, 1.5)
ymin, ymax = (0.25, 1.5)

κ = PeriodicKernel() + PolynomialKernel(1.0, 0.0, 1)
k(x,y) = kernel(κ, x, y)

x = [x for x in xmin:0.005:xmax]
y = [y for y in ymin:0.005:ymax]
Z = Float64[k(x,y) for y in y, x in x]

trace = heatmap(x=x, y=y, z=Z)

plot(trace)

