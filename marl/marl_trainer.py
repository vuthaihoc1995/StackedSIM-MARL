import torch


def train(env, actors, critic, optimizer_actor, optimizer_critic):
    obs = env.reset()

    log_probs = []
    rewards = []

    for _ in range(env.K):
        actions = []

        for k, actor in enumerate(actors):
            logits = actor(obs[k])
            idx, logp = actor.sample_action(logits)

            phi = idx * (2 * torch.pi / (2 ** actor.q_bits))
            actions.append(phi)

            log_probs.append(logp)

        obs, reward, done, info = env.step(actions)
        rewards.append(reward)

    # Return (scalar)
    G = sum(rewards)

    # Critic update
    state = torch.cat(obs).float()
    value = critic(state)

    loss_critic = (value.squeeze() - G) ** 2
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()

    # Actor update
    advantage = G - value.detach().squeeze()
    loss_actor = -torch.stack(log_probs).sum() * advantage

    optimizer_actor.zero_grad()
    loss_actor.backward()
    optimizer_actor.step()