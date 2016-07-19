using Gadfly, Colors

n = 60

x = [round(x,1) for x in vcat(2*(randn(n)-7), 2*(randn(n)+7), 2*randn(n))]
y = zeros(n*3)
c = vcat(Int64[1 for i = 1:2n], Int64[2 for i = 1:n])

P = plot(
        x=x,
        colour= map(class -> "Class $class", c),
        Geom.histogram(bincount=n),
        Scale.color_discrete_manual(colorant"red",colorant"blue"),
        Guide.XLabel("Feature"),
        Guide.YLabel(""),
        Guide.title("Non-Linearly Separable Data"),
        Guide.colorkey("")
    )

draw(PNG("Feature.png", 6inch, 3inch), P)

P = plot(
        x=x,
        y=x.^2,
        colour= map(class -> "Class $class", c),
        Geom.point(),
        Scale.color_discrete_manual(colorant"red",colorant"blue"),
        Guide.XLabel("Component 1"),
        Guide.YLabel("Component 2"),
        Guide.title("Linearly Separable Data"),
        Guide.colorkey(""),
        yintercept=[7^2], 
        Geom.hline(color=colorant"black")
    )

draw(PNG("FeatureMap.png", 6inch, 4inch), P)

