# Locked-run feedback

My read on the locked run is that the current result set is already coherent enough to support the paper and slide narrative without opening another broad optimization cycle.

## What looks strong already

- The **core RL story is clear**: once responsiveness is state-dependent and action history changes future reward, the task behaves like a true sequential control problem rather than a bandit.
- **H1 is strong** in the current figures. High separation clearly outperforms moderate and low separation in both convergence speed and final return.
- **H2 is also strong**, and the switching-cost plots reveal a useful tradeoff: higher switching cost reduces switching, but can also increase non-receptive occupancy.
- **H4 is clear** because the Value Iteration benchmark stays meaningfully above all model-free methods.
- **Expected SARSA looks like the strongest model-free baseline** in the locked run, with the smallest gap to optimal across the plotted settings.

## What still looks weaker / more nuanced

- **H3 remains mixed**. Double Q-Learning does not show a dramatic or universal stability advantage over Q-Learning in the current H=10, c_switch=0.0 figures.
- The current draft should treat H3 honestly as a nuanced result rather than overclaiming a win.

## Best use of the remaining polishing time

If there is more time later, the most valuable follow-up work is probably:

1. tightening the hypothesis-check language,
2. deciding how much of the H=5 material belongs in the main text versus appendix,
3. polishing captions / discussion,
4. making sure the final paper and deck align tightly with the grading rubric.

The locked run already gives a strong draft foundation.
