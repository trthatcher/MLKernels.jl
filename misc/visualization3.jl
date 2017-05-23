n = 100;

X = hcat(rand(n,2), zeros(n))
Y = [j == 3 ? X[i,1]*X[i,2] : X[i,j]^2 for i = 1:n, j = 1:3]


