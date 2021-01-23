from stable_baselines.common.policies import FeedForwardPolicy, register_policy


class DnnPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        layer_size = 20
        super(DnnPolicy, self).__init__(*args, **kwargs,
                                        net_arch=[dict(pi=[layer_size, layer_size, layer_size],
                                                       vf=[layer_size, layer_size, layer_size])],
                                        feature_extraction="mlp")
