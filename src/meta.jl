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

#=
function outer_constructor(obj::DataType)
    fields = fieldnames(obj)
    fieldtypes  = [field => fieldtype(obj, field).parameters[1].name for field in fields]
    arguments   = [:($field::Argument{$(fieldtypes[field])}) for field in fields]
    parameters  = [parameter.name for parameter in obj.parameters]
    constructor = Expr(:curly, obj.name.name, 
                       [Expr(:(<:), param.name, param.ub.name.name) 
                          for param in obj.parameters]...)
    definition_ls = Expr(:call, constructor, arguments...)
    definition_rs = Expr(:call, Expr(:curly, obj.name.name, parameters...), 
                         [Expr(:call, :convert, :(Variable{$(fieldtypes[field])}), field) 
                            for field in fields]...)
    Expr(:(=), definition_ls, definition_rs)
end
=#

function outer_constructor(obj::DataType, defaults::Tuple{Vararg{Real}})
    fields = fieldnames(obj)
    length(defaults) == length(fields) || error("wrong size")
    parameters  = [param.name => param.ub for param in obj.parameters]  # [:T=>Float64, :U=>Int]
    #println(parameters)
    fieldparams = Symbol[fieldtype(obj, field).parameters[1].name for field in fields]
    #println(fieldparams)
    first_idx   = [param => findfirst(fieldparams, param) for param in keys(parameters)]
    #println(first_idx)
    defaultparams = Symbol[if i == first_idx[fieldparams[i]]
                        get_default(parameters[fieldparams[i]]).name.name
                    else
                        fieldparams[i]
                    end for i in eachindex(fields)]

    arguments = [Expr(:(=), :($(fields[i])::Argument{$(fieldparams[i])}),
                            :(convert($(defaultparams[i]), $(defaults[i]))))
                 for i in eachindex(fields)]

    constructor = Expr(:curly, obj.name.name, 
                       [Expr(:(<:), param.name, param.ub.name.name) for param in obj.parameters]...)

    definition_ls = Expr(:call, constructor, arguments...)
end
    


#=
    targettypes = [i == findfirst(fieldtypes, fieldtype) ?
                       get_default(fieldtype(obj, fields[i]).ub) :
                       :(convert($(fieldtypes[i]), $(defaults[i])))
                   for i in eachindex(fieldtypes)]
    fieldfirst  = [i == findfirst(fieldtypes, fieldtype) for i in eachindex(fieldtypes)]
    arg_defaults = [fieldfirst[i] ?
                                  : 

                    for i in eachindex(fieldtypes)]
    arguments   = [Expr(:(=),
                        :($field::Argument{$(fieldtypes[i])}),
                        fieldfirst[fields[i]] == i ? :(convert(
                                                   : :(convert($(fieldtypes[i]), $(defaults[i]))

                        
                      for field in fields]
    parameters  = [parameter.name for parameter in obj.parameters]
    constructor = Expr(:curly, obj.name.name, 
                       [Expr(:(<:), param.name, param.ub.name.name) 
                          for param in obj.parameters]...)
    definition_ls = Expr(:call, constructor, arguments...)
    definition_rs = Expr(:call, Expr(:curly, obj.name.name, parameters...), 
                         [Expr(:call, :convert, :(Variable{$(fieldtypes[field])}), field) 
                            for field in fields]...)
    Expr(:(=), definition_ls, definition_rs)
end
=#
# MAke better!


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

#=

function outerconstructor(obj::DataType, T::DataType)
    T <: Real && !isleaftype(T) || error("Argument T type must be a real bits type")
    fields = fieldnames(obj)
    symobj = obj.name.name
    arguments = [:($arg::Variable{T}) for arg in fields]
    constructor_ls = Expr(:call, :($symobj{T<:$T}), arguments...)
    constructor_rs = Expr(:call, :($symobj{T}), fields...)
    Expr(:(=), constructor_ls, constructor_rs)
end

function outerconstructor(obj::DataType, T::DataType, n_T::Integer, U::DataType)
    T <: Real && !isleaftype(T) || error("Argument T type must be a real bits type")
    U <: Real && !isleaftype(U) || error("Argument U type must be a real bits type")
    fields = fieldnames(obj)
    n_T < length(fields) ||  error("Number of default arguments does not match fields")
    arguments = vcat([:($arg::Variable{T}) for arg in fields[1:n_T]],
                     [:($arg::Variable{U}) for arg in fields[n_T+1:end]])
    symobj = obj.name.name
    constructor_ls = Expr(:call, :($symobj{T<:$T,U<:$U}), arguments...)
    constructor_rs = Expr(:call, :($symobj{T,U}), fields...)
    Expr(:(=), constructor_ls, constructor_rs)
end

function outerconstructor(obj::DataType, T::DataType, T_args::Tuple{Vararg{Real}},
                                         U::DataType, U_args::Tuple{Vararg{Real}})
    T <: Real && isleaftype(T) || error("Default type T type must be a real bits type")
    U <: Real && isleaftype(U) || error("Default tye  U type must be a real bits type")
    n = length(T_args)
    m = length(U_args)
    fields = fieldnames(obj)
    length(fields) == (n + m) ||  error("Number of default arguments does not match fields")
    T_fields = fields[1:n]
    U_fields = fields[n+1:end]
    arguments_T = [Expr(:kw, :($(fields[i])::Variable), T_args[i]) for i = 1:n]
    arguments_U = [Expr(:kw, :($(fields[j])::Variable), U_args[j]) for j = 1:m]
    promotions_T = Expr(:call, :promote_arguments, T, fields_T...)
    promotions_U = Expr(:call, :promote_arguments, U, fields_U...)
    symobj = obj.name.name
        constructor_ls = Expr(:call, symkernel, arguments_T..., arguments_U...)
        constructor_rs = Expr(:call, symkernel, Expr(:..., promotions_T..., arguments_U...))
    Expr(:(=), constructor_ls, constructor_rs)
end

=#
