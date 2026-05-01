import torch


def train_sarl(env,
               actor,
               critic,
               optimizer_actor,
               optimizer_critic,
               T=5):
    """
    SARL trainer with T-step episodes

    env      : StackedRISEnv
    actor    : single Actor controlling all layers
    critic   : centralized Critic
    T        : number of steps per episode (>=5 recommended)
    """

    # --------------------------------------------------
    # Reset environment (NEW channel each episode)
    # --------------------------------------------------
    obs = env.reset()

    rewards = []
    log_probs = []

    # --------------------------------------------------
    # T-step episode
    # --------------------------------------------------
    for t in range(T):
        # --------------------------------------------------
        # Global observation (concatenate all layers)
        # --------------------------------------------------
        obs_sarl = torch.cat(obs)

        # Policy forward
        logits = actor(obs_sarl)

        # Sample discrete action indices
        idx, logp = actor.sample_action(logits)

        # Map indices -> physical RIS phases
        phi_all = idx * (2 * torch.pi / (2 ** actor.q_bits))

        # Split joint action into K layers
        actions = torch.split(phi_all, env.N)

        # Environment transition
        obs, reward, done, info = env.step(actions)

        rewards.append(reward)
        log_probs.append(logp)

    # --------------------------------------------------
    # Cumulative return
    # --------------------------------------------------
    G = sum(rewards)   # scalar

    # --------------------------------------------------
    # Critic update
    # --------------------------------------------------
    state = torch.cat(obs).float()
    value = critic(state).squeeze()

    loss_critic = (value - G) ** 2

    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # --------------------------------------------------
    # Actor update
    # --------------------------------------------------
    advantage = G - value.detach()

    loss_actor = -torch.stack(log_probs).sum() * advantage

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()

    return G