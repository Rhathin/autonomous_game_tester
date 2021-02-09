from stable_baselines.common.policies import FeedForwardPolicy


class DNNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        layer_size = 128
        super(DNNPolicy, self).__init__(*args, **kwargs,
                                        net_arch=[dict(pi=[256, layer_size, layer_size],
                                                       vf=[256, layer_size, layer_size])],
                                        feature_extraction="mlp")
