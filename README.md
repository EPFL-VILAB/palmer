# PALMER: Perception-Action Loop with Memory for Long-Horizon Planning
[Onur Beker](https://bekeronur.github.io/), [Mohammad Mohammadi](https://vilab.epfl.ch/), [Amir Zamir](https://vilab.epfl.ch/zamir/)

 [`Website`](https://palmer.epfl.ch/) | [`arXiv`](https://palmer.epfl.ch/) | [`BibTeX`](#citation)

![Experiment Visualizations](https://github.com/EPFL-VILAB/palmer/blob/main/palmer.gif)

<b><em><ins>TL;DR:</ins></em></b> We introduce PALMER, a <b>long-horizon planning method</b> that directly operates on high dimensional sensory input <b>observable by an agent on its own</b> (e.g., images from an onboard camera). Our key idea is to <b>retrieve previously observed trajectory segments</b> from a replay buffer and <b>restitch them into approximately optimal paths</b> to connect any given pair of start and goal states. This is achieved by combining <b>classical sampling-based planning algorithms</b> (e.g., PRM, RRT) with <b>learning-based perceptual representations</b> that are informed of actions and their consequences.[^1] 

[^1]: A more elaborate discussion around these motivations can be found in <a href="https://mitpress.mit.edu/9780262161831/vision-science/">[ref1,</a><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=sNaYnRcAAAAJ&citation_for_view=sNaYnRcAAAAJ:RGFaLdJalmkC">ref2]</a>.

## Summary
<p> To achieve autonomy in a priori unknown real-world scenarios, agents should be able to: 
<ol type="I">
<li>act directly from their own sensory observations, without assuming auxiliary instrumentation in their environment (e.g., a precomputed map, or an external mechanism to compute rewards).</li>
<li>learn from past experience to continually adapt and improve after deployment.</li>
<li>be capable of long-horizon planning.</li>
</ol>
</p>


Classical planning algorithms (e.g. PRM, RRT) are proficient at handling long-horizon planning. Deep learning based methods in turn can provide the necessary representations to address the others, by <b>modeling statistical contingencies between sensory observations</b>.[^2]

[^2]: For a conceptual discussion around statistical contingencies, please refer to <a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=81NhlCkAAAAJ&citation_for_view=81NhlCkAAAAJ:c1e4I3QdEKYC">[ref,</a><a href="https://en.wikipedia.org/wiki/Human_contingency_learning">wikipedia] </a>.


<p>
In this direction, we introduce a general-purpose planning algorithm called PALMER that combines <b>classical sampling-based planning algorithms</b> with <b>learning-based perceptual representations</b>. 
<ul>
<li>For <b>training</b> these representations, we <b>combine Q-learning with contrastive representation learning</b> to create a latent space where the distance between the embeddings of two states captures how easily an optimal policy can traverse between them. </li>
<li>For <b>planning</b> with these perceptual representations, we re-purpose classical sampling-based planning algorithms to <b>retrieve previously observed trajectory segments</b> from a replay buffer and <b>restitch them into approximately optimal paths</b> that connect any given pair of start and goal states. </li> 
</ul>
<p>
This creates a tight <b>feedback loop between representation learning, memory, reinforcement learning, and sampling-based planning</b>. The end result is <b>an experiential framework for long-horizon planning</b> that is <b>more robust and sample efficient</b> compared to existing methods.  
</p>
</p>

## Main Take-Aways
<ul>
<li><b>How to retrieve past trajectory segments from a replay-buffer / memory?</b> &#x2192 by using offline reinforcement learning for contrastive representation learning.</li>
 <li><b>How to restitch these trajectory segments into a new path?</b> &#x2192 by repurposing the main subroutines of classical sampling-based planning algorithms.</li>
<li><b>What makes PALMER robust and sample-efficient?</b> &#x2192 it explicitly checks back with a memory / training-dataset whenever it makes test-time decisions.</li>
</ul>

## How to Navigate this Codebase?
Please see [SETUP.md](SETUP.md) for instructions.

## Citation

```BibTeX
@article{beker2022palmer,
  author    = {Onur Beker and Mohammad Mohammadi and Amir Zamir},
  title     = {{PALMER}: Perception-Action Loop with Memory for Long-Horizon Planning},
  journal   = {arXiv preprint arXiv:coming soon!},
  year      = {2022},
}
```
