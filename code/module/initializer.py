import torch


# trunk model init
def default_weight_init(tensor):
    torch.nn.init.xavier_uniform(tensor)


def default_bias_init(tensor):
    torch.nn.init.constant_(tensor, 0)


# lite plugin model init
def default_lite_plugin_init(layer):
    torch.nn.init.xavier_uniform(layer.weight, gain=0.001)
    # torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)


# naive plugin model init
def default_naive_plugin_init(layer):
    torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)


# Initializers will be optimized in the future
#
# def weight_init_normal(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
#         torch.nn.init.normal_(m.weight, mean=0, std=0.01)
#     if isinstance(m, nn.Linear):
#         torch.nn.init.normal_(m.bias, mean=0, std=0.01)
#
#
# def weight_init_constant(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
#         torch.nn.init.constant_(m.weight, 0)
#     if isinstance(m, nn.Linear):
#         torch.nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # model.apply(weight_init_normal)
    dimension = 10
    plugin_layer = torch.nn.Linear(dimension, dimension // 2, True)
    print("-" * 50)
    print("original")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_weight_init(plugin_layer.weight)
    default_bias_init(plugin_layer.bias)
    print("-" * 50)
    print("trunk_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_lite_plugin_init(plugin_layer)
    print("-" * 50)
    print("lite_plugin_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)
    default_naive_plugin_init(plugin_layer)
    print("-" * 50)
    print("naive_plugin_init")
    print("plugin_layer.weight", plugin_layer.weight)
    print("plugin_layer.bias", plugin_layer.bias)