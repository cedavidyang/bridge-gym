# %%
import numpy as np
from bridge_gym.bridge_nbe107.rl_env import BridgeNBE107

from torchrl.envs.utils import check_env_specs
from torchrl.envs import GymWrapper

# %%
# test gym env

if __name__ == "__main__":
    n_element = 400
    max_steps, gamma = 50, 1/1.03

    env = BridgeNBE107(
        max_steps=max_steps, discount=gamma,
        n_element=n_element,
        include_step_count=True,
        reset_prob=np.array([1.0]+[0.0]*3, dtype=np.float32),
        # render_mode="ansi",
        render_mode="human",
        render_kwargs={'figsize': (6.5, 3.5)},
        seed=42,
        cost_kwargs={"normalizer": 1},
    )

    obs, _ = env.reset()
    env.render()

    obs_log = []
    action_log = []
    reward_log = []
    for _ in range(max_steps):
        # no action
        action = 0

        # custom action
        # if obs[2] > 0.3 or obs[3] > 0.05:
        #     action = 3
        # # elif obs[2] > 0.1:
        # #     action = 2
        # else:
        #     action = 0

        next_obs, reward, terminated, done, info = \
            env.step(action)

        env.render()

        obs_log.append(obs)
        action_log.append(action)
        reward_log.append(reward)
        obs = next_obs

# %%
# test wrapping to torchrl env (must run previous cell first)

if __name__ == "__main__":
    env.render_mode = "ansi"
    torch_env = GymWrapper(env, categorical_action_encoding=True)

    # integrity check
    print("Checking specs...")
    try:
        check_env_specs(torch_env)
        print("Check passed!")
    except Exception as e:
        print(f"Check failed: {e}")

    # test manual rollout
    td = torch_env.reset()
    print(f"Initial State Tensor: {td['observation']}")
    
    action = torch_env.action_spec.rand()
    td['action'] = action
    
    next_td = torch_env.step(td)

    print(f"Reward Tensor: {next_td['next', 'reward']}")
    print(f"Terminated: {next_td['next', 'terminated']}")
# %%
