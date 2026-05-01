import torch


def train(env,
          actors,
          critic,
          optimizer_actor,
          optimizer_critic,
          T=5):
    """
    MARL trainer with T-step episodes

    env      : StackedRISEnv
    actors   : list of K Actor networks
    critic   : centralized Critic
    T        : number of steps per episode (>=5 recommended)
    """

    # --------------------------------------------------
    # Reset environment (NEW channel each episode)
    # --------------------------------------------------
    obs = env.reset()

    log_probs = []
    rewards = []

    # --------------------------------------------------
    # T-step episode
    # --------------------------------------------------
    for t in range(T):
        actions = []

        # ----- Each layer (agent) selects its action -----
        for k, actor in enumerate(actors):
            # Forward policy
            logits = actor(obs[k])

            # Sample discrete phase indices
            idx, logp = actor.sample_action(logits)

            # Map index -> physical phase
            phi = idx * (2 * torch.pi / (2 ** actor.q_bits))

            actions.append(phi)
            log_probs.append(logp)

        # Environment transition
        obs, reward, done, info = env.step(actions)

        rewards.append(reward)

    # --------------------------------------------------
    # Cumulative return
    # --------------------------------------------------
    G = sum(rewards)   # scalar

    # --------------------------------------------------
    # Critic update (centralized)
    # --------------------------------------------------
    state = torch.cat(obs).float()
    value = critic(state).squeeze()

    loss_critic = (value - G) ** 2

    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # --------------------------------------------------
    # Actor update (policy gradient)
    # --------------------------------------------------
    advantage = G - value.detach()

    loss_actor = -torch.stack(log_probs).sum() * advantage

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()

    return G