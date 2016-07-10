function promote_type_float(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end

function promote_type_int(U_i::DataType...)
    U_max = promote_type(U_i...)
    U_max <: Signed ? U_max : Int64
end


# Checks to make sure the fields in the datatype are of type Parameter{T} or Parameter{U} where
# T<:AbstractFloat and U<:Integer
function fieldparameters(obj::DataType)
    fields = fieldnames(obj)
    0 < length(obj.parameters) <= 2 || error("Type must have one or two parameters (T & U)")
    for param in obj.parameters
        if param.name == :T
            param.ub == AbstractFloat || error("Parameter T must be T<:AbstractFloat")
        elseif param.name == :U
            param.ub == Integer || error("Parameter U must be U<:Integer")
        else
            error("Parameter $(param.name) not recognized; must be T<:AbstactFloat or U<:Integer")
        end
    end
    for field in fields
        field_type = fieldtype(obj, field)
        isa(field_type, DataType) || error("Field $field must be ::Parameter")
        if field_type.name.name != :HyperParameter 
            error("Fields must consist of Parameter types")
        end
        length(field_type.parameters) == 1 || error("Parameter type must have one type parameter")
        field_param = field_type.parameters[1]
        if field_param.name == :T
            if !(field_param.ub == AbstractFloat)
                error("Field $field must be ::Parameter{T<:AbstractFloat}")
            end
        elseif field_param.name == :U
            if !(field_param.ub == Integer)
                error("Field $field must be ::Parameter{U<:Integer}")
            end
        else
            error("$field type must be ::Parameter{T<:AbstractFloat} or ::Parameter{U<:Integer}")
        end
    end
    field_parameters = Symbol[fieldtype(obj, field).parameters[1].name for field in fields]
    return (fields, field_parameters)
end

function promote_code_block(obj::DataType, cstr_params, field_params)
    promote_T = Expr(:call, :promote_type_float, cstr_params[field_params .== :T]...)
    if length(obj.parameters) == 1
        return :(T = $promote_T)
    else
        promote_U = Expr(:call, :promote_type_int, cstr_params[field_params .== :U]...)
        return Expr(:block, :(T = $promote_T), :(U = $promote_U))
    end
end

function fieldparameters_constructor(obj::DataType)
    fields, field_params = fieldparameters(obj)
    n = length(fields)
    counter = [1,1]  # T count, U count
    constructor_params = Array(Symbol, n)
    constructor_types  = Array(Union{Symbol,Expr}, n)
    for i in eachindex(field_params)
        param_sym = field_params[i]
        param_idx = param_sym == :T ? 1 : 2
        constructor_params[i] = Symbol(string(param_sym, counter[param_idx]))
        if counter[param_idx] == 1
            constructor_types[i] = param_idx == 1 ? :Float64 : :Int64
        else
            preceding_params = (constructor_params[1:i-1])[field_params[1:i-1] .== param_sym]
            promotion = param_idx == 1 ? :promote_type_float : :promote_type_int
            constructor_types[i] = Expr(:call, promotion, preceding_params...)
        end
        counter[param_idx] += 1
    end
    (fields, field_params, constructor_params, constructor_types)
end

function generate_outer_constructor2(obj::DataType, default_values::Tuple{Vararg{Real}})
    # (:a,:b,:c,:d), [:T, :U, :T, :U], [:T1, :U1, :T2, :U2], 
    # [:Float64, :Int64, promote_type_float(T1), promote_type_int(U1)]
    fields, field_params, cstr_params, cstr_types = fieldparameters_constructor(obj)

    if (n = length(default_values)) != length(fields)
        error("Default count does not match field count")
    end

    # [:(T1 <: Real), :(T2 <: Real), :(U1 <: Real)
    defn_params = [Expr(:(<:), cstr_params[i], :Real) for i = 1:n]

    # (a::Argument{T1}=convert(Float64,x), b::Argument{T2}=convert(T1,x)...
    defn_args = [Expr(:kw, :($(fields[i])::Argument{$(cstr_params[i])}),
                           :(convert($(cstr_types[i]), $(default_values[i])))) for i = 1:n]

    # (Variable{T}(a), Variable{T}(b), Variable{U}(c))
    call_args = [:(Variable{$(field_params[i])}($(fields[i]))) for i = 1:n]

    block_definition = Expr(:call, Expr(:curly, obj.name.name, defn_params...), defn_args...)
    block_promotion  = promote_code_block(obj, cstr_params, field_params)
    block_call       = Expr(:call, Expr(:curly, obj.name.name, [p.name for p in obj.parameters]...), call_args...)

    return Expr(:function, block_definition, Expr(:block, block_promotion, block_call))
end


function generate_conversions(obj::DataType)
    fields, field_parameters = fieldparameters(obj)
    obj_sym = obj.name.name
    if length(obj.parameters) == 2
        # convert(::Type{obj{T,U}}, ::obj) = ...
        convert_target = Expr(:(::), Expr(:curly, :Type, Expr(:curly, obj_sym, :T, :U)))
        converted_arguments = 
            Expr[:(Variable(convert($(field_parameters[i]), obj.$(fields[i]).value), 
                   obj.$(fields[i]).isfixed)) for i in eachindex(fields)]
        conversion1 = Expr(:(=), Expr(:call, Expr(:curly, :convert, :T, :U), convert_target,
                                             Expr(:(::), :obj, obj_sym)),
                                 Expr(:call, obj_sym, converted_arguments...))
        # convert(::Type{obj{T}}, ::obj{_,U}) = ...
        convert_target = Expr(:(::), Expr(:curly, :Type, Expr(:curly, obj_sym, :T)))
        convert_source = Expr(:(::), Expr(:curly, :Type, Expr(:curly, obj_sym, :_, :U)))
        conversion2 = Expr(:(=), Expr(:call, Expr(:curly, :convert, :T, :_, :U), convert_target,
                                             Expr(:(::), :obj, Expr(:curly, obj_sym, :_, :U))),
                                 Expr(:call, obj_sym, converted_arguments...))
        return Expr(:block, conversion1, conversion2)
    elseif length(obj.parameters) == 1
        # convert(::Type{obj{T}}, ::obj) = ...
        convert_target      = Expr(:(::), Expr(:curly, :Type, Expr(:curly, obj_sym, :T)))
        converted_arguments = Expr[:(Variable(convert(T, obj.$(fields[i]).value), 
                                     obj.$(fields[i]).isfixed)) for i in eachindex(fields)]
        return Expr(:(=), Expr(:call, Expr(:curly, :convert, :T), convert_target,
                                      Expr(:(::), :obj, obj_sym)),
                          Expr(:call, obj_sym, converted_arguments...))
    else
        error("Data type should have 1 or 2 parameters")
    end
end

# Notes:
#   Assumes arguments are organized by type parameter
#   Assumes Variable{} is the argument type
function generate_outer_constructor(obj::DataType, defaults::Tuple{Vararg{Real}})
    fields, parameters = fieldparameters(obj)
    length(defaults) == length(fields) || error("Default count does not match field count")
    first_idx  = Dict(findfirst(parameters, :T) => :Float64, findfirst(parameters, :U) => :Int64)
    # Produces [:Float64, T, T, ..., Int64, U, U, ...]
    default_parameters = Symbol[get(first_idx, i, parameters[i]) for i in eachindex(parameters)]
    # Produces [arg1::Argument{T} = convert(Float64, default1), 
    #           arg2::Argument{T} = convert(T,default2)...]
    constructor_params = [Expr(:(<:), p.name, p.ub.name.name) for p in obj.parameters]
    constructor_args   = [Expr(:kw, :($(fields[i])::Argument{$(parameters[i])}),
                                    :(convert($(default_parameters[i]), $(defaults[i]))))
                          for i in eachindex(fields)]
    type_params = [p.name for p in obj.parameters]
    type_args   = [:(Variable($(fields[i]))) for i in eachindex(fields)]

    obj_sym = obj.name.name                           
    Expr(:(=), Expr(:call, Expr(:curly, obj_sym, constructor_params...), constructor_args...),
               Expr(:call, Expr(:curly, obj_sym, type_params...), type_args...))
end


macro outer_constructor(obj, defaults)
    eval(generate_outer_constructor2(eval(obj), eval(defaults)))
    eval(generate_conversions(eval(obj)))
end
