def get_parameters(model):
    total=0
    for parameter in list(model.parameters()):
        dim_size=1
        for dim in list(parameter.size()):
            dim_size = dim_size*dim
        total += dim_size
    return total