# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@CreateDate: 2023-04-05
@File: custom_multiagent
@Project: RLToys
"""
import os
import sys
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping
import time
import logging
import numpy as np
import functools
import random
from copy import copy
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils.env import ParallelEnv

_intro = """
For this tutorial, we will be creating a two-player game consisting of
a prisoner, trying to escape, and a guard, trying to catch the prisoner.
This game will be played on a 7x7 grid, where:

* The prisoner starts in the top left corner,
* the guard starts in the bottom right corner,
* the escape door is randomly placed in the middle of the grid, and
* both the prisoner and the guard can move in any of the four cardinal directions (up, down, left, right).
"""


class CustomEnvironment(ParallelEnv):
    """
    For this tutorial, we will be creating a two-player game consisting of
    a prisoner, trying to escape, and a guard, trying to catch the prisoner.
    This game will be played on a 7x7 grid, where:

    * The prisoner starts in the top left corner,
    * the guard starts in the bottom right corner,
    * the escape door is randomly placed in the middle of the grid, and
    * both the prisoner and the guard can move in any of the four cardinal directions (up, down, left, right).

    """
    def __init__(self, max_steps: int = 100):
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]
        self.max_steps = max_steps

    def reset(self, seed=None, return_info: bool = False, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # on a 7x7 grid
        self.prisoner_x = 0  # the top left corner
        self.prisoner_y = 0
        self.guard_x = 6  # the bottom right corner
        self.guard_y = 6

        # the escape door is randomly placed in the middle of the grid
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }
        return observations

    def step(self, actions):
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > self.max_steps:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        grid = np.empty((7, 7), dtype=str)
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "Door"
        # print(f"{grid} \n")
        return grid

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7 - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

    def get_action_masks(self):
        """Generate action masks"""
        prisoner_action_mask = np.ones(4, dtype=np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0  # Block left movement
        elif self.prisoner_x == 6:
            prisoner_action_mask[1] = 0  # Block right movement
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0  # Block down movement
        elif self.prisoner_y == 6:
            prisoner_action_mask[3] = 0  # Block up movement
        action_masks = {'prisoner': prisoner_action_mask}

        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == 6:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == 6:
            guard_action_mask[3] = 0

        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0
        action_masks['guard'] = guard_action_mask
        return action_masks

    def run(self, style='text'):
        if style == 'streamlit_arr':
            self.run_streamlit_arr()
        elif style == 'streamlit_heatmap':
            self.run_streamlit_heatmap()
        else:
            self.run_as_text()

    def run_as_text(self, num_iters=None):
        num_iters = self.max_steps if num_iters is None else num_iters
        observations = self.reset()
        logging.info(observations)

        for i in range(num_iters):
            print(f'Iter {i}'.center(40, '-'))
            action_masks = self.get_action_masks()
            actions = {agt: self.action_space(agt).sample(action_masks[agt]) for
                       agt in self.agents}
            observations, rewards, terminations, truncations, infos = self.step(
                actions=actions)
            time.sleep(0.5)
            print(f'observations: {observations}')
            print(f'rewards: {rewards}')
            print(f'terminations: {terminations}')
            print(f'terminations: {truncations}')
            grid = self.render()
            print(grid)
            if any(terminations.values()):
                break

    def run_streamlit_arr(self, num_iters=None):
        import streamlit as st
        num_iters = self.max_steps if num_iters is None else num_iters
        observations = self.reset()
        logging.info(observations)

        st.markdown('# 追逃游戏' + _intro)
        grid_show = st.empty()
        progress_bar = st.progress(0)
        iter_text = st.empty()
        info_show = st.empty()
        stop_but = st.button('Stop', key='is_stop')
        for i in range(num_iters):
            progress_bar.progress(i / num_iters)
            iter_text.text(f'Iter {i}'.center(40, '-'))
            action_masks = self.get_action_masks()
            actions = {agt: self.action_space(agt).sample(action_masks[agt]) for
                       agt in self.agents}
            observations, rewards, terminations, truncations, infos = self.step(
                actions=actions)
            time.sleep(0.5)
            info_show.text(
                f'observations: {observations}\n'
                f'rewards: {rewards}\n'
                f'terminations: {terminations}')
            grid = self.render()
            grid_show.dataframe(grid)
            is_stop = st.session_state['is_stop']
            if is_stop or any(terminations.values()):
                break
        st.button('Re-run')

    def run_streamlit_heatmap(self, num_iters=None):
        import streamlit as st
        from matplotlib import pyplot as plt
        import seaborn as sns
        num_iters = self.max_steps if num_iters is None else num_iters
        observations = self.reset()
        logging.info(observations)

        st.markdown('# 追逃游戏' + _intro)
        grid_show = st.empty()
        progress_bar = st.progress(0)
        iter_text = st.empty()
        info_show = st.empty()
        _ = st.button('Stop', key='is_stop')

        for i in range(num_iters):
            progress_bar.progress(i / num_iters)
            iter_text.text(f'Iter {i}'.center(40, '-'))
            action_masks = self.get_action_masks()
            actions = {agt: self.action_space(agt).sample(action_masks[agt]) for
                       agt in self.agents}
            observations, rewards, terminations, truncations, infos = self.step(
                actions=actions)
            time.sleep(0.5)
            info_show.text(
                f'observations: {observations}\n'
                f'rewards: {rewards}\n'
                f'terminations: {terminations}')
            grid = self.render()
            # grid_show.dataframe(grid)
            fig, ax = plt.subplots()
            grid_value = np.zeros(grid.shape, )
            grid_value[grid == 'G'] = - 1.
            grid_value[grid == 'P'] = 1.
            grid_value[grid == 'D'] = - 0.5
            sns.heatmap(grid_value, ax=ax,
                        linewidths=0.05, linecolor='grey',
                        annot=grid, cbar=False,
                        fmt='s', cmap='RdBu', vmax=1, vmin=-1)
            grid_show.pyplot(fig)
            if st.session_state['is_stop'] or any(terminations.values()):
                break
            plt.close(fig)
        st.button('Re-run')

