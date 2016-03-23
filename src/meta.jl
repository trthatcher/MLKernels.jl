function concrete_subtypes(T::DataType)
    ST = subtypes(T)
    if length(ST) > 0
        vcat(map(concrete_subtypes, ST)...)
    elseif !T.abstract # only collect concrete types
        [T]
    else
        []
    end
end

function supertypes(T::DataType)
    ancestors = DataType[]
    S = super(T)
    while S !== Any
        push!(ancestors, S)
        S = super(S)
    end
    ancestors
end

function promote_arguments(U::DataType, θ::Variable{Real}...)
    U <: Real && isbit(U) || error("Argument U type must be a real bits type")
    T = promote_type([eltype(x) for x in θ]...)
    T = T <: super(U) ? T : U
    tuple(Variable{T}[isa(x, Fixed) ? convert(Fixed{T}, x) : convert(T, x) for x in θ]...)
end

function get_default(obj::DataType)
    obj <: Real || error("Data type should be subtype of Real")
    obj <: Integer ? Int64 : Float64
end

function generate_outer_constructor(obj::DataType, defaults::Tuple{Vararg{Real}})
    fields = fieldnames(obj)
    length(defaults) == length(fields) || error("wrong size")
    parameters  = [param.name => param.ub for param in obj.parameters]  # [:T=>Float64, :U=>Int]
    fieldparams = Symbol[fieldtype(obj, field).parameters[1].name for field in fields]
    first_idx   = [param => findfirst(fieldparams, param) for param in keys(parameters)]

    defaultparams = Symbol[if i == first_idx[fieldparams[i]]
                        get_default(parameters[fieldparams[i]]).name.name
                    else
                        fieldparams[i]
                    end for i in eachindex(fields)]

    arguments = [Expr(:kw, :($(fields[i])::Argument{$(fieldparams[i])}),
                            :(convert($(defaultparams[i]), $(defaults[i]))))
                 for i in eachindex(fields)]

    constructor = Expr(:curly, obj.name.name, 
                       [Expr(:(<:), param.name, param.ub.name.name) for param in obj.parameters]...)

    definition_ls = Expr(:call, constructor, arguments...)

    definition_rs = Expr(:call, 
                         Expr(:curly, obj.name.name, 
                              [param.name for param in obj.parameters]...), 
                         [Expr(:call, :convert, :(Variable{$(fieldparams[i])}), fields[i]) 
                          for i in eachindex(fields)]...)

    Expr(:(=), definition_ls, definition_rs)
end

#=
function generate_conversion(obj::DataType)
    fields = fieldnames(obj)
    length(defaults) == length(fields) || error("wrong size")
    parameters  = Symbol[param.name for param in obj.parameters]
    fieldparams = Symbol[fieldtype(obj, field).parameters[1].name for field in fields]

        # Stopped here.
    defaultparams = Symbol[if i == first_idx[fieldparams[i]]
                        get_default(parameters[fieldparams[i]]).name.name
                    else
                        fieldparams[i]
                    end for i in eachindex(fields)]

    arguments = [Expr(:kw, :($(fields[i])::Argument{$(fieldparams[i])}),
                            :(convert($(defaultparams[i]), $(defaults[i]))))
                 for i in eachindex(fields)]

    constructor = Expr(:curly, obj.name.name, 
                       [Expr(:(<:), param.name, param.ub.name.name) for param in obj.parameters]...)

    definition_ls = Expr(:call, constructor, arguments...)

    definition_rs = Expr(:call, 
                         Expr(:curly, obj.name.name, 
                              [param.name for param in obj.parameters]...), 
                         [Expr(:call, :convert, :(Variable{$(fieldparams[i])}), fields[i]) 
                          for i in eachindex(fields)]...)

    Expr(:(=), definition_ls, definition_rs)
end
=#
    
macro outer_constructor(obj, defaults)
    eval(generate_outer_constructor(eval(obj), eval(defaults)))
end




#=
function generic_constructor(obj::DataType, defaults::Tuple{Vararg{Real}})
    fields = fieldnames(obj)
    fieldtypes = [field => fieldtype(obj, field).parameters[1] for field in fields]
    arguments = [Expr(:kw, :($(fields[i])::Argument), defaults[i]) for i in eachindex(fields)]
    conversions = [:(convert(Variable{$(get_default(fieldtypes[field].ub))}, $field))
                   for field in fields]
    definition_ls = Expr(:call, obj.name.name, arguments...)
    definition_rs = Expr(:call, obj.name.name, conversions...)
    Expr(:(=), definition_ls, definition_rs)
end

function kernel_constructors(obj::DataType, defaults::Tuple{Vararg{Real}})
    Expr(:block, outer_constructor(eval(obj)), generic_constructor(eval(obj), defaults))
end
=#
