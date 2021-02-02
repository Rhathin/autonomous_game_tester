from stable_baselines.common.policies import FeedForwardPolicy


class DnnPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        layer_size = 128
        super(DnnPolicy, self).__init__(*args, **kwargs,
                                        net_arch=[dict(pi=[256, layer_size, layer_size],
                                                       vf=[256, layer_size, layer_size])],
                                        feature_extraction="mlp")
