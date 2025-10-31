import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_reward_and_length(base_stats_df, imp1_stats_df,imp2_stats_df, outdir):
    """Plots the reward and episode length curves for both runs."""
    
    print("Plotting training rewards and episode length...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(base_stats_df['generation'], base_stats_df['avg_unshaped_r'], label='Base (Avg Unshaped)', color='C0', linestyle=':')
    ax1.plot(imp1_stats_df['generation'], imp1_stats_df['avg_unshaped_r'], label='Improved 1 (Avg Unshaped)', color='C0')
    ax1.plot(imp2_stats_df['generation'], imp2_stats_df['avg_unshaped_r'], label='Improved 2 (Avg Unshaped)', color='C0',linestyle='--')
    
    ax1.plot(imp1_stats_df['generation'], imp1_stats_df['avg_shaped_r'], label='Improved 1 (Avg Shaped)', color='C1')
    ax1.plot(imp2_stats_df['generation'], imp2_stats_df['avg_shaped_r'], label='Improved 2 (Avg Shaped)', color='C1',linestyle='--')
    
    ax1.set_title('Average Reward per Generation')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(base_stats_df['generation'], base_stats_df['avg_length'], label='Base (Avg Length)', color='C2', linestyle=':')
    ax2.plot(imp1_stats_df['generation'], imp1_stats_df['avg_length'], label='Improved 1 (Avg Length)', color='C2')
    ax2.plot(imp2_stats_df['generation'], imp2_stats_df['avg_length'], label='Improved 2 (Avg Length)', color='C2',linestyle='--')
    
    ax2.set_title('Average Episode Length per Generation')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Steps')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(outdir, 'comparison_training_curves.png')
    plt.savefig(save_path)
    print(f"Saved training curve plot to {save_path}")
    plt.close()

def plot_final_achievements(ach_df, title, filename):
    """Plots a bar chart for the final generation's achievement unlock rates."""
    
    print(f"Plotting final achievements for: {title}")
    
    last_gen = ach_df['generation'].max()
    
    final_gen_ach = ach_df[ach_df['generation'] == last_gen].copy()
    
    final_gen_ach['unlock_rate_pct'] = final_gen_ach['unlock_rate'] * 100.0
    
    final_gen_ach = final_gen_ach.sort_values(by='unlock_rate_pct', ascending=False)
    
    if final_gen_ach.empty:
        print(f"No achievement data found for '{title}'. Skipping plot.")
        return

    plt.figure(figsize=(10, 7))
    plt.bar(final_gen_ach['achievement_name'], final_gen_ach['unlock_rate_pct'])
    
    plt.title(f'{title} (Generation {last_gen})')
    plt.ylabel('Unlock Rate (%)')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(outdir, filename)
    plt.savefig(save_path)
    print(f"Saved achievement plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison of two training runs.")
    parser.add_argument('--base_dir', help='Path to the BASE log directory (e.g., logdir/crafter_ga_base/final)', default="logdir/crafter_ga_base/final")
    parser.add_argument('--imp1_dir', help='Path to the IMPROVED 1 log directory (e.g., logdir/crafter_ga_imp2/final)', default="logdir/crafter_ga_imp2/final")
    parser.add_argument('--imp2_dir', help='Path to the IMPROVED 2 log directory (e.g., logdir/crafter_ga_imp3/final)', default="logdir/crafter_ga_imp3/final")
    parser.add_argument('--outdir', default='logdir/plots', help='Directory to save the plots')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    try:
        base_stats_df = pd.read_csv(os.path.join(args.base_dir, 'generation_stats.csv'))
        base_ach_df = pd.read_csv(os.path.join(args.base_dir, 'achievement_stats.csv'))
        
        imp1_stats_df = pd.read_csv(os.path.join(args.imp1_dir, 'generation_stats.csv'))
        imp1_ach_df = pd.read_csv(os.path.join(args.imp1_dir, 'achievement_stats.csv'))

        imp2_stats_df = pd.read_csv(os.path.join(args.imp2_dir, 'generation_stats.csv'))
        imp2_ach_df = pd.read_csv(os.path.join(args.imp2_dir, 'achievement_stats.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file. Make sure paths are correct.")
        print(e)
        exit(1)

    plot_reward_and_length(base_stats_df, imp1_stats_df,imp2_stats_df, outdir)

    plot_final_achievements(base_ach_df, 'Base Run - Final Achievements', 'base_final_achievements.png')
    plot_final_achievements(imp1_ach_df, 'Improved Run 1 - Final Achievements', 'improved1_final_achievements.png')
    plot_final_achievements(imp2_ach_df, 'Improved Run 2 - Final Achievements', 'improved2_final_achievements.png')
    
    print(f"\nAll plots generated successfully. Saved in {args.outdir}")
