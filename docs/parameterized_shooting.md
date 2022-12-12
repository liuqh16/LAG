# Parameterized Policy for Shooting

A natural form of policy parameterization for shooting is to have a policy network, which outputs the parameter of a Bernoulli distribution. Let $\phi$ denote the parameters of the policy network, the action $a$ for a given state $s$ is sampled as

$$
a \sim \operatorname{Bern}(p), p=f_\phi(s) .
$$

The main drawback of this approach is that the policy will have a high possibility to explore in the meaningless region, e.g., the target is well out of its reachable distance. To reduce the ineffective exploration, we propose a human priori to the policy. In particular, we choose the Beta distribution as the priori and train the network outputs as the "likelihood". The prior is given by a predefined rule, such as

$$
\operatorname{Beta}\left(\alpha_0, \beta_0\right), \alpha_0, \beta_0=h(s) .
$$

Suppose that the network outputs are $\alpha, \beta=f_\phi(s)$, the final policy distribution (the posterior) will be $\operatorname{Beta}\left(\alpha_0+\alpha, \beta_0+\beta\right)$. To control the confidence level of likelihood, we force the parameters of posteriori within $[0, C]$. In other word, $\alpha \in\left[0, C-\alpha_0\right]$ and $\beta \in\left[0, C-\beta_0\right]$.