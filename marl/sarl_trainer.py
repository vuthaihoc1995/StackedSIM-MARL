import torch


def train_sarl(env, actor, critic, optimizer_actor, optimizer_critic):
    obs = env.reset()

    rewards = []

    # Global observation
    obs_sarl = torch.cat(obs)

    # ----- Action selection (no gradient) -----
    with torch.no_grad():
        logits = actor(obs_sarl)
        idx, _ = actor.sample_action(logits)

    # Recompute logits for policy gradient
    logits = actor(obs_sarl)
    _, logp = actor.sample_action(logits)

    # Decode joint action → per-layer actions
    phi_all = idx * (2 * torch.pi / (2 ** actor.q_bits))
    actions = torch.split(phi_all, env.N)

    obs, reward, done, info = env.step(actions)
    rewards.append(reward)

    # Return
    G = sum(rewards)

    # ----- Critic update -----
    state = torch.cat(obs).float()
    value = critic(state)

    loss_critic = (value.squeeze() - G) ** 2
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # ----- Actor update -----
    advantage = G - value.detach().squeeze()
    loss_actor = -logp * advantage

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()