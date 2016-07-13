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
function checkfields(obj::DataType)
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
    return fields
end

function fieldparameters(obj::DataType)
    fields = checkfields(obj)
    field_parameters = Symbol[fieldtype(obj, field).parameters[1].name for field in fields]
    return (fields, field_parameters)
end

# [:T, :U, :T, :U] -> ([:T1, :U1, :T2, :U2], [:Float64, :Int64, promote_type_float(T1), ...]
function constructorparameters(field_params)
    n = length(field_params)
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
    (constructor_params, constructor_types)
end

function generate_outer_constructor(obj::DataType, default_values::Tuple{Vararg{Real}})
    # (:a,:b,:c,:d), [:T, :U, :T, :U], 
    fields, field_params = fieldparameters(obj)

    # [:T1, :U1, :T2, :U2], [:Float64, :Int64, promote_type_float(T1), promote_type_int(U1)]
    cstr_params, cstr_types = constructorparameters(field_params)

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

    sym_obj = obj.name.name

    ex_definition = Expr(:call, Expr(:curly, sym_obj, defn_params...), defn_args...)
    ex_body = Expr(:block)

    ex1 = Expr(:call, :promote_type_float, cstr_params[field_params .== :T]...)
    push!(ex_body.args, :(T = $ex1))

    if length(obj.parameters) == 2
        ex2 = Expr(:call, :promote_type_int, cstr_params[field_params .== :U]...)
        push!(ex_body.args, :(U = $ex2))
    end

    ex3 = Expr(:call, Expr(:curly, sym_obj, [p.name for p in obj.parameters]...), call_args...)
    push!(ex_body.args, ex3)
   
    return Expr(:function, ex_definition, ex_body)
end

function generate_conversion_TU(obj::DataType)
    fields, field_params = fieldparameters(obj)
    sym_obj = obj.name.name
    
    ex1 = :(convert{T,U}(::Type{$sym_obj{T,U}}, f::$sym_obj))
    ex2 = :(convert{T,_,U}(::Type{$sym_obj{T}}, f::$sym_obj{_,U}))
    
    args = [:(Variable{$(field_params[i])}(f.$(fields[i]))) for i in eachindex(fields)]

    ex3 = Expr(:call, :($sym_obj{T,U}), args...)

    return Expr(:block, Expr(:(=), ex1, ex3), Expr(:(=), ex2, ex3))
end

function generate_conversion_T(obj::DataType)
    fields = checkfields(obj)
    sym_obj = obj.name.name
    
    ex1 = :(convert{T}(::Type{$sym_obj{T}}, f::$sym_obj))

    if length(fields) == 0
        return Expr(:(=), ex1, Expr(:call, :($sym_obj{T}))) 
    else
        args = [:(Variable{T}(f.$(fields[i]))) for i in eachindex(fields)]
        return Expr(:(=), ex1, Expr(:call, :($sym_obj{T}), args...))
    end
end

function generate_conversions(obj::DataType)
    length(obj.parameters) == 2 ? generate_conversion_TU(obj) : generate_conversion_T(obj)
end

macro outer_constructor(obj, defaults)
    eval(generate_outer_constructor(eval(obj), eval(defaults)))
    eval(generate_conversions(eval(obj)))
end
