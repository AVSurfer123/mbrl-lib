import math

import gym
import numpy as np
from gym import logger, spaces
from gym.utils import seeding
from feedback_rl.splines import ConstAccelSpline, Spline

from mbrl.env.cartpole_continuous import CartPoleEnv


class CartPoleEnvAcc(gym.Env):
    # This is a continuous version of gym's cartpole environment, with the only difference
    # being valid actions are any numbers in the range [-1, 1], and the are applied as
    # a multiplicative factor to the total force.
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, spline_time_horizon=20):
        self.env = CartPoleEnv()

        self.time_horizon = spline_time_horizon
        self.timestep = spline_time_horizon * self.env.tau
        self.pole_inertia_coefficient = 1
        self.known_states = 2

        self.counter = 0
        self.curr_action = 0
        self.spline = None
        self.xi_initial = None
        self.force = 0

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = self.env.observation_space

    def seed(self, seed=None):
        self.env.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_once(self, action=None):
        """
        Provide an action. Then can provide None for next self.time_horizon - 1 calls to step the inner environment once.
        """
        if action is None:
            if self.counter >= self.time_horizon:
                raise ValueError("Cannot step anymore for current acceleration")
        else:
            self.counter = 0

            # Get scalar action
            if isinstance(action, list):
                action = action[0]
            else:
                action = action.squeeze()

            self.curr_action = action
            self.spline = ConstAccelSpline(num_knots=2, init_vel=self.env.state[1])
            self.spline.set_spline([0, self.timestep], [action])
            self.xi_initial = np.array([self.env.state[0], self.env.state[1]])

        self.force = feedback_controller(self.env, self.counter * self.env.tau, self.spline, self.xi_initial)
        obs, rew, done, info = self.env.step(self.force)     

        self.counter += 1

        return obs, rew, done, info

    def step(self, action):
        reward = 0

        for i in range(self.time_horizon):
            inner_action = action if i == 0 else None

            obs, rew, done, _  = self.step_once(inner_action)
            reward += rew
            if done:
                break

        return np.array(self.env.state), reward, done, {}

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()

    def xi_dynamics(self, obs, action):
        spline = ConstAccelSpline(num_knots=2)
        spline.set_spline([0, self.timestep], action)
        xi_initial = np.array([obs[0], obs[1]])
        next_xi = xi_initial + spline.x(self.timestep)
        return next_xi

def feedback_controller(env: CartPoleEnv, t, traj: Spline, xi_initial: np.ndarray):
    g = 9.81
    M = env.masscart
    m = env.masspole
    l = env.length * 2
    theta = env.state[2]
    theta_dot = env.state[3]
    xi = np.array([env.state[0], env.state[1]])

    x_des = traj.deriv(t, 0)
    x_dot_des = traj.deriv(t, 1)
    xi_des = np.array([x_des + xi_initial[0], x_dot_des]) # + xi_initial # Adding xi_initial since we assume spline value is an offset from starting position
    v_ff = traj.deriv(t, 2)

    # Choose closed-loop eigenvalues to be -3, -3, using standard CCF dynamics
    K = np.array([-900, -60])
    v = v_ff + K @ (xi - xi_des)

    u_star = -m*l*np.sin(theta) * theta_dot**2  - m*g*np.sin(theta)*np.cos(theta) / (1 + env.pole_inertia_coefficient)
    F = u_star + (M + m - m*np.cos(theta)**2 / (1 + env.pole_inertia_coefficient)) * v
    return F

def test_controller():
    from mbrl.env.cartpole_continuous import CartPoleEnv
    import time
    import matplotlib.pyplot as plt
    FPS = 30
    env = CartPoleEnv()
    # env.seed()
    obs = env.reset()
    num_knots = 5
    traj_des = ConstAccelSpline(num_knots, env.state[1])
    STEPS = 20 * num_knots
    times = np.linspace(0, STEPS, num_knots) * env.tau
    # knots = times * np.sin(times) / 4
    # traj_des.build_spline(times, knots)
    param = np.random.uniform(-1, 1, size=(num_knots - 1,))
    print("Random param:", param)
    traj_des.set_spline(times, param)
    data = [obs]
    xi_initial = np.array([env.state[0], env.state[1]])
    for i in range(STEPS):
        action = feedback_controller(env, i * env.tau, traj_des, xi_initial)
        print("Action:", action)
        obs, rew, done, info = env.step(action)
        print("obs:", obs)
        data.append(obs)
        env.render()
        time.sleep(1/FPS)
        if done:
            print("Done at", i)
            break
    data = np.array(data)
    print(data.shape)
    
    ax = plt.gca()
    traj_des.plot(ax, order=0)
    traj_des.plot(ax, order=1)
    traj_des.plot(ax, order=2)
    eval_times = np.arange(0, i + 2) * env.tau
    diff = data[:, 0] - xi_initial[0]
    print(diff)
    plt.plot(eval_times, diff, label='x actual')
    plt.plot(eval_times, data[:, 1], label='x dot actual')
    plt.legend()
    plt.savefig('tracking_sine_offset.png')
    plt.show()

if __name__ == '__main__':
    # test_controller()
    # quit()
    env = CartPoleEnvAcc()
    obs = env.reset()
    data = [obs]
    splines = []
    initial = []
    import time
    import matplotlib.pyplot as plt
    FPS = 30
    STEPS = 5
    num_steps = 0
    for i in range(STEPS):
        action = np.random.uniform(-1, 1, (1,))
        print("Action:", action)

        for j in range(env.time_horizon):
            inner_action = action if j == 0 else None

            obs, rew, done, _  = env.step_once(inner_action)

            num_steps += 1
            data.append(obs)
            env.render()
            time.sleep(1/FPS)

            if done:
                print("Done at stage", j)
                break

        splines.append(env.spline)
        initial.append(env.xi_initial[0])

        if done:
            print("Done at spline", i)
            break

    data = np.array(data)
    
    ax = plt.gca()
    eval_times = np.arange(0, num_steps + 1) * env.env.tau
    for i in range(len(splines)):
        times = np.linspace(0, env.timestep, num=500)
        points = splines[i].eval_spline(times, 0)

        ax.plot(times + i * env.timestep, points + initial[i], label=f"Spline {i} $x$")
        ax.legend()
        ax.grid()
        ax.set(xlabel='Time', ylabel='Value', title='Spline Path')

    plt.plot(eval_times, data[:, 0], label='x actual')
    plt.plot(eval_times, data[:, 1], label='x dot actual')
    plt.legend()
    plt.savefig('tracking_controller.png')
    plt.show()
