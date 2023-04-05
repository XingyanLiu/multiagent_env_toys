# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@CreateDate: 2023-04-05
@File: custom_multiagent_v0
@Project: RLToys
"""
import os
import sys
from pathlib import Path
import time
import logging
import numpy as np
from gymnasium.core import ActType


def __test__streamlit():
    import streamlit as st
    from env.guard_prisoner import CustomEnvironment, _intro
    from matplotlib import pyplot as plt
    import seaborn as sns
    env = CustomEnvironment(max_steps=200)
    observations = env.reset()
    logging.info(observations)

    st.markdown('# 追逃游戏' + _intro)
    grid_show = st.empty()
    progress_bar = st.progress(0)
    iter_text = st.empty()
    info_show = st.empty()
    total_iters = env.max_steps
    for i in range(total_iters):
        progress_bar.progress(i / total_iters)
        iter_text.text(f'Iter {i}'.center(40, '-'))
        action_masks = env.get_action_masks()
        actions = {agt: env.action_space(agt).sample(action_masks[agt]) for agt in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions=actions)
        time.sleep(0.5)
        info_show.text(
            f'observations: {observations}\n'
            f'rewards: {rewards}\n'
            f'terminations: {terminations}')
        grid = env.render()
        # grid_show.dataframe(grid)
        fig, ax = plt.subplots()
        grid_value = np.zeros(grid.shape, dtype=int)
        grid_value[grid == 'G'] = - 1
        grid_value[grid == 'P'] = 1
        sns.heatmap(grid_value, ax=ax,
                    linewidths=0.05, linecolor='grey',
                    annot=grid, cbar=False,
                    fmt='s', cmap='RdBu', vmax=1, vmin=-1)
        grid_show.pyplot(fig)
        if any(terminations.values()):
            break
        plt.close(fig)


def main():
    """
    use ``streamlit run xxx.py`` to launch it
    """
    from env.guard_prisoner import CustomEnvironment
    env = CustomEnvironment()
    # env.run_streamlit_arr()
    env.run_streamlit_heatmap()


def __test__():
    from env.guard_prisoner import CustomEnvironment
    env = CustomEnvironment(max_steps=20)
    env.run_as_text()
    # from pettingzoo.test import parallel_api_test
    # parallel_api_test(CustomEnvironment(), num_cycles=50)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    t = time.time()

    # __test__streamlit()
    # __test__()
    main()

    print('Done running file: {}\nTime: {}'.format(
        os.path.abspath(__file__), time.time() - t,
    ))
