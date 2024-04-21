import torch


def diffusion_dynamics(model, betas, init):
    """Diffusion dynamics (reverse process decoder).

    Args:
      model: Diffusion probabilistic network.
      betas: Noise schedule.
      init: Initial state for Langevin dynamics (usually Gaussian noise).

    Returns:
      state: Final state sampled from Langevin dynamics.
      collection: Array of state at each step of sampling with shape
          (num_sigmas * T + 1 + int(denoise), :).
      ld_metrics: Metrics collected for each noise level with shape (num_sigmas, T).
    """
    infill_samples = torch.zeros(init.shape)
    infill_masks = torch.zeros(init.shape)

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_prev = torch.stack([torch.ones((1,)), alphas_prod[:-1]])
    assert alphas.shape == alphas_prod.shape == alphas_prod_prev.shape

    # collection_steps = 40
    # start = init * (1 - infill_masks) + infill_samples * infill_masks
    # images = torch.zeros((collection_steps + 1, *init.shape))
    # collection = jax.ops.index_update(images, jax.ops.index[0, :], start)
    # collection_idx = torch.linspace(1, len(betas), collection_steps).int()

    def sample_with_beta(current, t):
        state = current

        # Noise schedule constants
        beta = betas[t]
        alpha = alphas[t]
        alpha_prod = alphas_prod[t]
        alpha_prod_prev = alphas_prod_prev[t]

        # Constants for posterior q(x_t|x_0)
        sqrt_reciprocal_alpha_prod = torch.sqrt(1 / alpha_prod)
        sqrt_alpha_prod_m1 = torch.sqrt(1 - alpha_prod) * sqrt_reciprocal_alpha_prod

        # Create infilling template
        infill_noise_cond = t > 0
        infill_noise = torch.randn(infill_samples.shape)
        noisy_y = (
            torch.sqrt(alpha_prod) * infill_samples + torch.sqrt(1 - alpha_prod) * infill_noise
        )
        y = infill_noise_cond * noisy_y + (1 - infill_noise_cond) * infill_samples

        # Constants for posterior q(x_t-1|x_t, x_0)
        posterior_mu1 = beta * torch.sqrt(alpha_prod_prev) / (1 - alpha_prod)
        posterior_mu2 = (1 - alpha_prod_prev) * torch.sqrt(alpha) / (1 - alpha_prod)

        # Clipped variance (must be non-zero)
        posterior_var = beta * (1 - alpha_prod_prev) / (1 - alpha_prod)
        posterior_var_clipped = torch.maximum(posterior_var, torch.tensor([1e-20]))
        posterior_log_var = torch.log(posterior_var_clipped)

        # Noise
        noise_cond = t > 0
        noise = torch.randn(state.shape)
        noise = noise_cond * noise + (1 - noise_cond) * torch.zeros(state.shape)
        noise = noise * torch.exp(0.5 * posterior_log_var)

        # Reverse process (reconstruction)
        noise_condition_vec = torch.sqrt(alpha_prod) * torch.ones((noise.shape[0], 1))
        noise_condition_vec = noise_condition_vec.reshape(
            init.shape[0], *([1] * len(init.shape[1:]))
        )
        eps_recon: torch.Tensor = model(state, noise_condition_vec)
        state_recon = sqrt_reciprocal_alpha_prod * state - sqrt_alpha_prod_m1 * eps_recon
        state_recon = torch.clip(state_recon, -1.0, 1.0)
        posterior_mu = posterior_mu1 * state_recon + posterior_mu2 * state
        next_state = posterior_mu + noise

        # Infill
        next_state = next_state * (1 - infill_masks) + y * infill_masks

        # Collect metrics
        step = state - next_state
        grad_norm = torch.norm(step, p=2, dim=1).mean()
        noise_norm = torch.norm(noise, p=2, dim=1).mean()
        step_norm = torch.norm(step, p=2, dim=1).mean()
        curr_metrics = (grad_norm, step_norm, alpha_prod, noise_norm)

        return next_state, curr_metrics

    # init_params = (init, rng, collection)
    curr_state = init
    samples = [curr_state]
    _metrics = torch.zeros(len(betas), 4)
    for _t in range(len(betas)):
        curr_state, metrics = sample_with_beta(curr_state, _t)
        samples.append(curr_state)
        _metrics[_t] = metrics

    return samples, _metrics


if __name__ == "__main__":
    pass
