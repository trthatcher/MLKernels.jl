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
        if field_type.name.name != :Parameter 
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
        # combined codeblock
        return quote
            $conversion1
            $conversion2
        end
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
    type_args   = [:(convert(Variable{$(parameters[i])}, $(fields[i]))) for i in eachindex(fields)]

    obj_sym = obj.name.name                           
    Expr(:(=), Expr(:call, Expr(:curly, obj_sym, constructor_params...), constructor_args...),
               Expr(:call, Expr(:curly, obj_sym, type_params...), type_args...))
end

macro outer_constructor(obj, defaults)
    eval(generate_outer_constructor(eval(obj), eval(defaults)))
    eval(generate_conversions(eval(obj)))
end
