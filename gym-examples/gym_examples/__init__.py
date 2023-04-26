from gym.envs.registration import register

register(
    id="gym_examples/GridWorld-v3",
    entry_point="gym_examples.envs:GridWorldEnv",
)
