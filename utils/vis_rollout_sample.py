import numpy as np
import matplotlib.pyplot as plt

min_num_blocks = 21
max_num_blocks = 42
decay_steps = 1000
initial_bias_exponent = 3.0
num_samples = 100000
steps_to_plot = [0, 250, 500, 750, 1000]

samples_by_step = {}

for step in steps_to_plot:
    progress = min(1.0, step / decay_steps)
    exponent = 1.0 + (initial_bias_exponent - 1.0) * (1.0 - progress)

    rand = np.random.rand(num_samples)
    weighted_rand = rand ** exponent

    range_size = max_num_blocks - min_num_blocks + 1
    num_generated_blocks = min_num_blocks + np.floor(weighted_rand * range_size).astype(int)
    num_generated_blocks = np.clip(num_generated_blocks, min_num_blocks, max_num_blocks)

    samples_by_step[step] = num_generated_blocks

fig, ax = plt.subplots(figsize=(10, 6))

bins = np.arange(min_num_blocks - 0.5, max_num_blocks + 1.5, 1) # Bin edges for integer counts

for step in steps_to_plot:
    samples = samples_by_step[step]
    counts, _ = np.histogram(samples, bins=bins)
    total_count = len(samples)
    probabilities = counts / total_count
    
    x_positions = np.arange(min_num_blocks, max_num_blocks + 1)
    
    ax.plot(x_positions, probabilities, marker='o', linestyle='-', label=f'step={step}', alpha=0.7)

ax.set_xlabel('Num Generated Blocks')
ax.set_ylabel('Probability')
ax.set_title('Probability Distribution of Num Generated Blocks (Exponential Decay)')
ax.set_xticks(np.arange(min_num_blocks, max_num_blocks + 1))
ax.legend()
ax.grid(axis='y', linestyle='--')
plt.tight_layout()

file_name = f'min_{min_num_blocks}_max_{max_num_blocks}_decay_step_{decay_steps}_init_bias_{initial_bias_exponent}_rollout_sample_distribution_curve.png'
plt.savefig(file_name)