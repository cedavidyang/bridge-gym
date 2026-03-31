import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float32

# gynasium imports
import gymnasium as gym
from gymnasium import spaces

# import element costs
from .settings import NCS, NA
from .settings import CS_PFS, FAILURE_COST
from .settings import ACTION_MODEL, UNIT_COSTS
from .cost_util import normalized_cost as cost_util


class BridgeNBE107(gym.Env):
    # standard: should not change
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self, max_steps, discount, n_element,
        state_size=NCS, action_size=NA,
        include_step_count=False,
        reset_prob: Float32[np.ndarray, f"{NCS}"] = np.array([1.0]+[0.0]*(NCS-1), dtype=np.float32),
        action_model=ACTION_MODEL,
        unit_costs=UNIT_COSTS,
        pf_array=CS_PFS,
        failure_cost=FAILURE_COST,
        render_mode=None,
        render_kwargs: dict | None = None,
        cost_kwargs: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        # store env parameters
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.n_element = n_element
        self.include_step_count = include_step_count

        self.discount = discount
        self.pf_array = pf_array
        self.failure_cost = failure_cost
        self.action_model, self.unit_costs = action_model, unit_costs
        self.reset_prob = reset_prob

        if cost_kwargs is None:
            self.cost_kwargs = {}
        else:
            self.cost_kwargs = cost_kwargs

        # define observation and action spaces
        obs_size = self.state_size+1 if include_step_count else self.state_size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_size)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._seed = seed
        self._first_rest = True

        # plotting parameters
        self.fig, self.ax, self.colors = None, None, None
        self.render_kwargs = {} if render_kwargs is None else render_kwargs
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        # set seed
        if seed is not None:
            super().reset(seed=seed)
        elif self._first_rest:
            # only initialize if needed
            super().reset(seed=self._seed)
            self._first_rest = False

        self._time = 0
        self._state = self.reset_prob

        self._cs_map = self.np_random.choice(
            size=self.n_element,
            a=np.arange(self.state_size),
            p=self.reset_prob,
        )

        if self.include_step_count:
            observation = np.append(self._state, self._time/self.max_steps)
        else:
            observation = self._state
        info = {"cs": self._state, "time": self._time}

        return observation, info

    def step(self, action):

        # reward due to action
        state0 = self._state
        dir_cost = self.unit_costs[int(action)] @ (state0*self.n_element)
        fail_risk = (self.pf_array @ state0) * (self.failure_cost*self.n_element)
        cost = dir_cost + fail_risk
        reward = cost_util(cost, **self.cost_kwargs)
        discount_factor = self.discount**self._time
        reward = (discount_factor*reward).astype(np.float32)

        # update hidden state
        self._update_state(action)
        self._time += 1

        # update observation
        if self.include_step_count:
            observation = np.append(self._state, self._time/self.max_steps)
        else:
            observation = self._state

        # check if done
        if self._time > self.max_steps:
            done = True
            _, info = self.reset()
        else:
            done = False
            info = {
                "cs": state0, "next_cs": self._state,
                "dir_cost": dir_cost, "fail_risk": fail_risk,
                "cost": cost, "reward": reward, "discount": discount_factor
            }

        terminated = done

        return observation, reward, terminated, done, info

    def render(self):
        if self.render_mode == "human":
            self._render_gui()

        elif self.render_mode == "ansi":
            print(f"Step {self._time}: CS = {self._state}")

    def close(self):
        pass

    def _render_gui(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1, tight_layout=True, **self.render_kwargs)
            self.colors = plt.cm.viridis(np.linspace(0, 1, self.state_size))
            self.ax.set_xlim(-0.8, self.max_steps+0.8)
            self.ax.set_ylim(0, 1.05)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("CS distribution")
            # setup colorbar
            norm = plt.Normalize(vmin=0, vmax=self.state_size - 1)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
            cbar = self.fig.colorbar(
                mappable, 
                ax=self.ax, 
                ticks=range(self.state_size),
                boundaries=np.arange(-0.5, self.state_size, 1) # Centers ticks in color blocks
            )
            cbar.set_ticklabels([f'CS{i+1}' for i in range(self.state_size)])
            plt.show(block=False)

        # Draw the stacked bar for the current time step
        bottom = 0
        for i in range(self.state_size):
            val = self._state[i]
            self.ax.bar(
                self._time, val, bottom=bottom,
                color=self.colors[i], width=0.8,
            )
            bottom += val

        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_state(self, action):
        Tmtx = self.action_model[int(action)]

        # update cs map
        for i in range(self.n_element):
            # for element i, random select based on the transition matrix
            pi = Tmtx[self._cs_map[i], :]
            self._cs_map[i] = self.np_random.choice(
                size=1,
                a=np.arange(self.state_size),
                p=pi,
            )
        
        # update state based on different cs counts in cs_map
        state = np.bincount(self._cs_map, minlength=self.state_size) / self.n_element
        self._state = state