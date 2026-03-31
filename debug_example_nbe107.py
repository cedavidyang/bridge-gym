# %%
import numpy as np
from bridge_gym.example_nbe107.rl_env import SingleElement

from torchrl.envs.utils import check_env_specs
from torchrl.envs import GymWrapper


# %%
# test gym env

if __name__ == "__main__":
    max_steps, gamma = 50, 1/1.03
    reset_prob = np.array([1.0]+[0.0]*3, dtype=np.float32)
    alpha_vector = None
    # reset_prob = None
    # alpha_vector = np.array([0.14964171, 0.11136174, 0.05003725, 0.03926025])

    env = SingleElement(
        max_steps=max_steps, discount=gamma,
        include_step_count=True,
        reset_prob=reset_prob,
        dirichlet_alpha=alpha_vector,
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
        # action = 0

        # custom action
        if obs[3] > 0.05:
            action = 3
        else:
            action = 0

        next_obs, reward, terminated, done, info = \
            env.step(action)

        env.render()

        obs_log.append(obs)
        action_log.append(action)
        reward_log.append(reward)
        obs = next_obs
    
    # reformat the figure using env.fig
    # change figuze size to 3.25 by 2.5
    env.fig.set_size_inches(3.25, 2.0)

    # set x label

    # change font size to 9
    font_size = 8
    for ax in env.fig.axes:
        # Update axis labels
        if ax.xaxis.label:
            ax.xaxis.label.set_size(font_size)
        if ax.yaxis.label:
            ax.yaxis.label.set_size(font_size)

        # Update tick labels
        ax.tick_params(axis='x', which='major', labelsize=font_size)
        ax.tick_params(axis='y', which='major', labelsize=font_size)

    # change x label
    env.fig.axes[0].set_xlabel("Time (years)", fontsize=font_size)

    # clean up the layout so the newly enlarged text doesn't overlap
    env.fig.tight_layout()

    # env.fig.savefig("../figures/plots/cs_do_nothing.png", dpi=600)
    # env.fig.savefig("../figures/plots/cs_custom1.png", dpi=600)

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
