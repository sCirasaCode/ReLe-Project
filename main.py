from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os


class HospitalRLExperiment:
    """
    Reinforcement Learning Experiment für BEAR Hospital Environment
    """
    
    def __init__(self, building_type="Hospital", climate="Hot_Humid", location="Denver"):
        self.building_type = building_type
        self.climate = climate
        self.location = location
        self.env = None
        self.rl_model = None
        
    def create_environment(self):
        """Erstelle BEAR Hospital Environment"""
        params = ParameterGenerator(self.building_type, self.climate, self.location)
        self.env = BuildingEnvReal(params)
        print(f"🏥 Environment erstellt: {self.building_type} in {self.location}")
        print(f"📊 Observation Space: {self.env.observation_space.shape}")
        print(f"🎮 Action Space: {self.env.action_space.shape}")
        return self.env
    
    def train_rl_agent(self, total_timesteps=50_000, algorithm="PPO"):
        """Trainiere RL-Agent"""
        if self.env is None:
            self.create_environment()
            
        # Vectorized Environment für Stable-Baselines3
        vec_env = make_vec_env(lambda: self.env, n_envs=1)
        
        # PPO Agent
        self.rl_model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        print(f"🚀 Starte {algorithm} Training ({total_timesteps:,} steps)...")
        self.rl_model.learn(total_timesteps=total_timesteps)
        print("✅ Training abgeschlossen!")
        
        return self.rl_model
    
    def evaluate_rl_agent(self, num_episodes=1, episode_length=24):
        """Evaluiere den trainierten RL-Agent"""
        if self.rl_model is None:
            raise ValueError("Kein trainierter Agent vorhanden!")
            
        results = {
            'states': [],
            'actions': [],
            'rewards': [],
            'temperatures': [],
            'energy_consumption': []
        }
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_states, episode_actions, episode_rewards = [], [], []
            
            for step in range(episode_length):
                action, _ = self.rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_states.append(obs.copy())
                episode_actions.append(action.copy())
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            results['states'].append(np.array(episode_states))
            results['actions'].append(np.array(episode_actions))
            results['rewards'].append(np.array(episode_rewards))
        
        return results
    
    def evaluate_random_baseline(self, num_episodes=1, episode_length=24):
        """Evaluiere Random Baseline für Vergleich"""
        results = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_states, episode_actions, episode_rewards = [], [], []
            
            for step in range(episode_length):
                action = self.env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_states.append(obs.copy())
                episode_actions.append(action.copy())
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            results['states'].append(np.array(episode_states))
            results['actions'].append(np.array(episode_actions))
            results['rewards'].append(np.array(episode_rewards))
        
        return results
    
    def analyze_temperature_control(self, results, zone_indices=[0, 1, 2]):
        """Analysiere Temperaturkontrolle für spezifische Zonen"""
        states = results['states'][0]  # Erste Episode
        
        # Extrahiere Temperaturen (angenommen erste N Werte sind Temperaturen)
        temperatures = {}
        for zone_idx in zone_indices:
            if zone_idx < states.shape[1]:
                temperatures[f'Zone_{zone_idx}'] = states[:, zone_idx]
        
        return temperatures
    
    def visualize_results(self, rl_results, random_results):
        """Visualisiere Ergebnisse: RL vs Random"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Temperaturverläufe (erste 3 Zonen)
        ax1 = axes[0, 0]
        rl_temps = self.analyze_temperature_control(rl_results, [0, 1, 2])
        random_temps = self.analyze_temperature_control(random_results, [0, 1, 2])
        
        timesteps = np.arange(len(rl_results['rewards'][0]))
        
        for zone, temps in rl_temps.items():
            ax1.plot(timesteps, temps, label=f'RL {zone}', linewidth=2)
        
        for zone, temps in random_temps.items():
            ax1.plot(timesteps, temps, '--', alpha=0.7, label=f'Random {zone}')
        
        ax1.axhline(22, color='black', linestyle=':', alpha=0.5, label='Setpoint 22°C')
        ax1.axhline(22.5, color='red', linestyle=':', alpha=0.5)
        ax1.axhline(21.5, color='red', linestyle=':', alpha=0.5)
        ax1.set_title('Temperaturverläufe (24h)')
        ax1.set_xlabel('Stunden')
        ax1.set_ylabel('Temperatur [°C]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Kumulative Rewards
        ax2 = axes[0, 1]
        rl_rewards = rl_results['rewards'][0]
        random_rewards = random_results['rewards'][0]
        
        ax2.plot(timesteps, np.cumsum(rl_rewards), label='RL Agent', linewidth=2)
        ax2.plot(timesteps, np.cumsum(random_rewards), '--', label='Random Baseline')
        ax2.set_title('Kumulative Rewards (Kosten)')
        ax2.set_xlabel('Stunden')
        ax2.set_ylabel('Kumulative Rewards')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Reward Distribution
        ax3 = axes[1, 0]
        ax3.hist(rl_rewards, bins=20, alpha=0.7, label='RL Agent', density=True)
        ax3.hist(random_rewards, bins=20, alpha=0.7, label='Random Baseline', density=True)
        ax3.set_title('Reward Verteilung')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Dichte')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metriken
        ax4 = axes[1, 1]
        metrics = {
            'Total Reward': [np.sum(rl_rewards), np.sum(random_rewards)],
            'Mean Reward': [np.mean(rl_rewards), np.mean(random_rewards)],
            'Reward Std': [np.std(rl_rewards), np.std(random_rewards)]
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        rl_values = [metrics[key][0] for key in metrics.keys()]
        random_values = [metrics[key][1] for key in metrics.keys()]
        
        ax4.bar(x - width/2, rl_values, width, label='RL Agent')
        ax4.bar(x + width/2, random_values, width, label='Random Baseline')
        ax4.set_title('Performance Vergleich')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics.keys(), rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results_experiment.png", dpi=300)
        
        # Print Summary
        print("\n📊 EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"RL Agent - Total Reward: {np.sum(rl_rewards):.2f}")
        print(f"Random Baseline - Total Reward: {np.sum(random_rewards):.2f}")
        print(f"Improvement: {((np.sum(rl_rewards) - np.sum(random_rewards)) / abs(np.sum(random_rewards)) * 100):.1f}%")


def main():
    """Hauptexperiment"""
    # Experiment Setup
    experiment = HospitalRLExperiment(
        building_type="Hospital",
        climate="Hot_Humid", 
        location="Denver"
    )
    
    # 1. Environment erstellen
    experiment.create_environment()
    
    # 2. RL Agent trainieren
    experiment.train_rl_agent(total_timesteps=30_000)
    
    # 3. Evaluation
    print("\n🔬 Evaluiere RL Agent...")
    rl_results = experiment.evaluate_rl_agent(num_episodes=1, episode_length=24)
    
    print("🎲 Evaluiere Random Baseline...")
    random_results = experiment.evaluate_random_baseline(num_episodes=1, episode_length=24)
    
    # 4. Visualisierung
    print("📈 Erstelle Visualisierungen...")
    experiment.visualize_results(rl_results, random_results)


if __name__ == "__main__":
    main()