import gym

class NormalizedEnv(gym.Wrapper):

    def _reset(self):
        observation = self.env.reset()
        return self._observation(observation)

    def _step(self, action):
        action = self._action(action)
        observation, reward, done, info = self.env.step(action)
        return self._observation(observation), reward, done, info

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _observation(self, observation):
        obs_k_inv = 2./(self.observation_space.high - self.observation_space.low)
        obs_b = (self.observation_space.high + self.observation_space.low)/ 2.
        return obs_k_inv * (observation - obs_b)

