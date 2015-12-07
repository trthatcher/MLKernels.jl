A  = [1 2 3;
      4 5 6;
      7 8 9]
AL = [1 2 3;
      2 5 6;
      3 6 9]
AU = [1 4 7;
      4 5 8;
      7 8 9]
U = [3 3 3;
     0 3 3;
     0 0 3]
L = [3 0 0;
     3 3 0;
     3 3 3]

dotc = sum(A .* A, 1)
dotr = sum(A .* A, 2)

w = [1; 2; 3]

wdotc = sum((A .* A) .* w , 1)
wdotr = sum((A .* A) .* w', 2)

info("Testing ", MLKernels.syml)
for T in FloatingPointTypes 
    @test MLKernels.syml(convert(Array{T},A)) == convert(Array{T}, AL)
end

info("Testing ", MLKernels.symu)
for T in FloatingPointTypes
    @test MLKernels.symu(convert(Array{T},A)) == convert(Array{T}, AU) 
end

info("Testing ", MLKernels.dot_columns)
for T in FloatingPointTypes 
    @test MLKernels.dot_columns(convert(Array{T},A)) == vec(convert(Array{T}, dotc))
    @test MLKernels.dot_columns(convert(Array{T},A), convert(Array{T},w)) == vec(convert(Array{T}, wdotc))
end

info("Testing ", MLKernels.dot_rows)
for T in FloatingPointTypes 
    @test MLKernels.dot_rows(convert(Array{T},A)) == vec(convert(Array{T}, dotr))
    @test MLKernels.dot_rows(convert(Array{T},A), convert(Array{T},w)) == vec(convert(Array{T}, wdotr))
end

info("Testing ", MLKernels.matrix_prod!)
for T in FloatingPointTypes
    @test MLKernels.matrix_prod!(convert(Array{T},A), convert(Array{T},A)) == convert(Array{T}, A .* A)
    @test MLKernels.matrix_prod!(convert(Array{T},U), convert(Array{T},U), true, false) == 3*convert(Array{T},U)
    @test MLKernels.matrix_prod!(convert(Array{T},U), convert(Array{T},U), true, true)  == 9*ones(T, size(U)...)
    @test MLKernels.matrix_prod!(convert(Array{T},L), convert(Array{T},L), false, false) == 3*convert(Array{T},L)
    @test MLKernels.matrix_prod!(convert(Array{T},L), convert(Array{T},L), false, true)  == 9*ones(T, size(L)...)
end

info("Testing ", MLKernels.matrix_sum!)
for T in FloatingPointTypes
    @test MLKernels.matrix_sum!(convert(Array{T},A), convert(Array{T},A)) == convert(Array{T}, A .+ A)
    @test MLKernels.matrix_sum!(convert(Array{T},U), convert(Array{T},U), true, false) == 2*convert(Array{T},U)
    @test MLKernels.matrix_sum!(convert(Array{T},U), convert(Array{T},U), true, true)  == 6*ones(T, size(U)...)
    @test MLKernels.matrix_sum!(convert(Array{T},L), convert(Array{T},L), false, false) == 2*convert(Array{T},L)
    @test MLKernels.matrix_sum!(convert(Array{T},L), convert(Array{T},L), false, true)  == 6*ones(T, size(L)...)

end

A = [1 2;
     3 4;
     5 6]

b = [1;
     2]

c = [1;
     2;
     3]

info("Testing ", MLKernels.translate!)
for T in FloatingPointTypes
    @test MLKernels.translate!(convert(Array{T},A), one(T)) == convert(Array{T}, A .+ one(T))
    @test MLKernels.translate!(one(T), convert(Array{T},A)) == convert(Array{T}, A .+ one(T))
    @test MLKernels.translate!(convert(Array{T},A), convert(Array{T},b)) == convert(Array{T}, A .+ b')
    @test MLKernels.translate!(convert(Array{T},c), convert(Array{T},A)) == convert(Array{T}, A .+ c)
end
