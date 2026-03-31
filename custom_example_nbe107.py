# %%
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from bridge_gym.example_nbe107.rl_env import SingleElement
from bridge_gym.example_nbe107.settings import CS_PFS

# %%

def action_policy(obs):

    # DP policy
    # cs = np.argmax(obs)
    # if cs == 0:
    #     action = 1
    # elif cs == 1:
    #     action = 2
    # elif cs == 2:
    #     action = 2
    # elif cs == 3:
    #     action = 3

    # GA policy
    pf = obs @ CS_PFS
    beta = stats.norm.ppf(1-pf)
    
    if beta > 3.558:
        action = 1
    elif beta > 3.367:
        action = 2
    elif beta > 3.191:
        action = 3
    else:
        action = 4

    # oblique tree policy (without human)
    # if -5.72*obs[0] + 0.663*obs[1] + 3.88 > 0:
    #     action = 2
    # else:
    #     action = 1

    # oblique tree policy (with human)
    # if obs[3] > 0.1:
    #     action = 3
    # elif -5.72*obs[0] + 0.663*obs[1] + 3.88 > 0:
    #     action = 2
    # else:
    #     action = 1

    return action


if __name__ == "__main__":
    num_episodes = 1_000
    seed = 42

    max_steps, gamma = 200, 1/1.03
    include_step_count = False

    reset_prob = None
    alpha_vector = np.array([0.14964171, 0.11136174, 0.05003725, 0.03926025])

    env = SingleElement(
        max_steps=max_steps, discount=gamma,
        include_step_count=False,
        reset_prob=reset_prob,
        dirichlet_alpha=alpha_vector,
        render_mode="ansi",
        # render_mode="human",
        render_kwargs={'figsize': (6.5, 3.5)},
        seed=seed,
        cost_kwargs={"normalizer": 1},
    )

    total_reward_log = np.zeros(num_episodes)
    init_stats_log = []
    init_beta_log = np.zeros(num_episodes)

    for i in range(num_episodes):
        obs, _ = env.reset()

        if include_step_count:
            states = obs[:-1]
            init_stats_log.append(states)
        else:
            states = obs
            init_stats_log.append(states)
        
        init_pf = states @ CS_PFS
        init_beta = -stats.norm.ppf(init_pf)
        init_beta_log[i] = init_beta

        obs_log = []
        action_log = []
        reward_log = []
        for _ in range(max_steps):
            # custom policy
            action = action_policy(obs)

            next_obs, reward, terminated, done, info = \
                env.step(action)

            obs_log.append(obs)
            action_log.append(action)
            reward_log.append(reward)

            obs = next_obs
        
        total_reward = sum(reward_log)
        total_reward_log[i] = total_reward

    print(f"Average reward: {total_reward_log.mean()}")
    print(f"Std reward: {total_reward_log.std()}")

    with sns.plotting_context("notebook", font_scale=1.0):
        sns.set_style('ticks')
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        sns.scatterplot(x=init_beta_log, y=-total_reward_log, ax=ax)   

# %%