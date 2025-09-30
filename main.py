from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Controller.MPC_Controller import MPCAgent
from BEAR.Utils.utils_building import ParameterGenerator
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import os


# ============================================================
# Custom Reward Wrapper (für besseres RL-Training)
# ============================================================
class CustomRewardWrapper(gym.Wrapper):
    """
    Wrappt BEAR und ersetzt Reward:
    r = -β * ||action||^2 - (1-β) * ||TempDeviation||^2
    """
    def __init__(self, env, beta=0.5, comfort_band=0.5, setpoint=22.0):
        super().__init__(env)
        self.beta = beta
        self.comfort_band = comfort_band
        self.setpoint = setpoint

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Temperaturabweichung (kritische Zonen: z.B. Zone 10 = OR1)
        critical_zones = [10, 17, 21]  # OR1, ICU, Patientenzimmer
        deviations = []
        for zone_idx in critical_zones:
            if zone_idx < len(obs):
                temp = obs[zone_idx]
                deviation = max(0, abs(temp - self.setpoint) - self.comfort_band)
                deviations.append(deviation)
        
        avg_deviation = np.mean(deviations) if deviations else 0

        # Custom Reward
        energy_cost = np.linalg.norm(action) ** 2
        comfort_cost = avg_deviation ** 2
        new_reward = -(self.beta * energy_cost + (1 - self.beta) * comfort_cost)

        return obs, new_reward, terminated, truncated, info


# ============================================================
# SAC Training & Evaluation
# ============================================================
def train_sac_agent(building='Hospital', climate='Hot_Humid', location='Denver', 
                    timesteps=10_000, beta=0.5, save_path="models"):
    """
    Trainiert einen SAC-Agent für Gebäudesteuerung
    
    Args:
        building: Gebäudetyp (z.B. 'Hospital')
        climate: Klimazone (z.B. 'Hot_Humid')
        location: Standort (z.B. 'Denver')
        timesteps: Anzahl Trainingsschritte
        beta: Gewichtung Energie vs Komfort (0-1)
        save_path: Ordner zum Speichern des Modells
    
    Returns:
        Trainiertes SAC-Modell
    """
    os.makedirs(save_path, exist_ok=True)
    
    print("="*60)
    print("🤖 SAC TRAINING")
    print("="*60)
    print(f"Gebäude: {building}")
    print(f"Klima: {climate}, Standort: {location}")
    print(f"Trainingsschritte: {timesteps:,}")
    print(f"Beta (Energie/Komfort): {beta}")
    print("="*60 + "\n")
    
    # Environment erstellen
    def make_env():
        params = ParameterGenerator(building, climate, location)
        env = BuildingEnvReal(params)
        env = CustomRewardWrapper(env, beta=beta)
        return env
    
    # Vectorized Environment
    vec_env = make_vec_env(make_env, n_envs=1)
    
    # SAC Agent mit optimierten Hyperparametern
    model = SAC(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',  # Automatische Entropie-Anpassung
        target_update_interval=1,
        target_entropy='auto'
    )
    
    print("🚀 Starte SAC Training...")
    model.learn(total_timesteps=timesteps)
    
    # Modell speichern
    model_path = os.path.join(save_path, f"sac_{building}_{location}.zip")
    model.save(model_path)
    print(f"\n✅ Training abgeschlossen!")
    print(f"💾 Modell gespeichert: {model_path}\n")
    
    return model


def evaluate_sac_agent(model, building='Hospital', climate='Hot_Humid', 
                       location='Denver', hours=24, beta=0.5):
    """
    Evaluiert einen trainierten SAC-Agent
    
    Args:
        model: Trainiertes SAC-Modell
        building, climate, location: Umgebungsparameter
        hours: Simulationsdauer in Stunden
        beta: Reward-Gewichtung
    
    Returns:
        states, actions, rewards (numpy arrays)
    """
    print("🔬 Evaluiere SAC Agent...")
    
    # Environment erstellen
    params = ParameterGenerator(building, climate, location)
    env = BuildingEnvReal(params)
    env = CustomRewardWrapper(env, beta=beta)
    
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    
    for t in range(hours):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        states.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    print(f"✅ SAC Evaluation abgeschlossen (Total Reward: {np.sum(rewards):.2f})\n")
    
    return np.array(states), np.array(actions), np.array(rewards)


# ============================================================
# Visualisierung & Statistiken
# ============================================================
def plot_temperature_profiles(states, zone_names, hours=24, agent_name="Agent", save_path="results"):
    """ Visualisiert Temperaturprofile für Schlüsselzonen über einen bestimmten Zeitraum. """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))

    """ Ausgewählte kritische Zonen """
    key_zones = {
        'ICU_FLR_2': 17,
        'OR1_FLR_2': 10,
        'PATROOM_MULTI10_FLR_3': 21,
        'LOBBY_RECORDS_FLR_1': 7,
        'BASEMENT': 0
    }

    for name, idx in key_zones.items():
        temperatures = [state[idx] for state in states[:hours]]
        ax.plot(range(hours), temperatures, label=name, linewidth=2)
    
    # Komfortband einzeichnen
    ax.axhline(22, color='black', linestyle='--', linewidth=1.5, label='Setpoint 22°C')
    ax.axhline(22.5, color='red', linestyle=':', alpha=0.5, label='±0.5°C Band')
    ax.axhline(21.5, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Stunden')
    ax.set_ylabel('Temperatur (°C)')
    ax.set_title(f'Temperaturprofile der kritischen Zonen über 24 Stunden ({agent_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f"temperature_profiles_{agent_name}.png")
    plt.savefig(filepath, dpi=300)
    print(f"📊 Temperaturprofil gespeichert: {filepath}")
    plt.close()


def plot_energy_consumption(actions, hours=24, agent_name="Agent", save_path="results"):
    """ Visualisiert den Energieverbrauch über einen bestimmten Zeitraum. """
    os.makedirs(save_path, exist_ok=True)
    
    total_energy = []
    for i in range(hours):
        """ Berechnung des Energieverbrauchs als Summe der absoluten Aktionswerte """
        energy = sum(abs(action) for action in actions[i])
        total_energy.append(energy)
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(hours), total_energy, marker='o', linewidth=2, color='steelblue')
    plt.xlabel('Stunden')
    plt.ylabel('Energieverbrauch (kWh)')
    plt.title(f'Energieverbrauch über 24 Stunden ({agent_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(save_path, f"energy_consumption_{agent_name}.png")
    plt.savefig(filepath, dpi=300)
    print(f"📊 Energieverbrauch gespeichert: {filepath}")
    plt.close()


def calculate_statistics(states, actions, agent_name="Agent"):
    """ Berechnet und druckt statistische Kennzahlen für Temperaturen und Energieverbrauch. """
    temperatures = np.array(states)
    actions = np.array(actions)

    temp_mean = np.mean(temperatures, axis=0)
    temp_std = np.std(temperatures, axis=0)
    energy_mean = np.mean(np.sum(np.abs(actions), axis=1))
    energy_std = np.std(np.sum(np.abs(actions), axis=1))

    print(f"\n{'='*60}")
    print(f"📊 STATISTIKEN ({agent_name})")
    print(f"{'='*60}")
    
    # Nur kritische Zonen anzeigen
    critical_zones = {
        'BASEMENT': 0,
        'LOBBY_RECORDS_FLR_1': 7,
        'OR1_FLR_2': 10,
        'ICU_FLR_2': 17,
        'PATROOM_MULTI10_FLR_3': 21
    }
    
    print("Temperaturstatistiken (kritische Zonen):")
    for name, idx in critical_zones.items():
        print(f"{name:30s}: {temp_mean[idx]:.2f}°C ± {temp_std[idx]:.2f}°C")
    
    print(f"\nEnergieverbrauch (Mittelwert ± Standardabweichung): {energy_mean:.2f} kWh ± {energy_std:.2f} kWh")
    print(f"{'='*60}\n")


def compare_agents(results_dict, save_path="results"):
    """
    Vergleicht mehrere Agenten (MPC, SAC, etc.)
    
    Args:
        results_dict: Dict mit Format {"Agent_Name": (states, actions, rewards)}
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = {"MPC": "green", "SAC": "blue", "PPO": "orange", "Random": "red"}
    
    # 1. Temperaturvergleich (Zone 10 = OR1)
    ax1 = axes[0]
    for name, (states, _, _) in results_dict.items():
        temps = [state[10] for state in states[:24]]
        ax1.plot(range(24), temps, label=name, linewidth=2, 
                color=colors.get(name, "gray"))
    
    ax1.axhline(22, color='black', linestyle='--', label='Setpoint')
    ax1.axhline(22.5, color='red', linestyle=':', alpha=0.5)
    ax1.axhline(21.5, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Stunden')
    ax1.set_ylabel('Temperatur (°C)')
    ax1.set_title('Temperaturvergleich OR1 (Zone 10)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Kumulative Rewards
    ax2 = axes[1]
    for name, (_, _, rewards) in results_dict.items():
        ax2.plot(range(len(rewards)), np.cumsum(rewards), label=name, 
                linewidth=2, color=colors.get(name, "gray"))
    
    ax2.set_xlabel('Stunden')
    ax2.set_ylabel('Kumulative Rewards')
    ax2.set_title('Kostenvergleich (höher = besser)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, "agent_comparison.png")
    plt.savefig(filepath, dpi=300)
    print(f"📊 Vergleich gespeichert: {filepath}")
    plt.close()


# ============================================================
# Main mit SAC + MPC
# ============================================================
def main():
    num_hours = 24
    save_path = "results"
    beta = 0.5  # 50/50 Balance zwischen Energie und Komfort
    
    # ========== SAC Training ==========
    sac_model = train_sac_agent(
        building='Hospital',
        climate='Hot_Humid',
        location='Denver',
        timesteps=10_000,
    )
    
    # ========== SAC Evaluation ==========
    states_sac, actions_sac, rewards_sac = evaluate_sac_agent(
        sac_model,
        building='Hospital',
        climate='Hot_Humid',
        location='Denver',
        hours=num_hours,
        beta=beta
    )
    
    # ========== MPC Evaluation ==========
    print("🎯 Evaluiere MPC Agent...")
    Parameter = ParameterGenerator('Hospital', 'Hot_Humid', 'Denver')
    env_mpc = BuildingEnvReal(Parameter)
    env_mpc.reset()
    
    mpc_agent = MPCAgent(env_mpc, gamma=env_mpc.gamma, safety_margin=0.96, planning_steps=10)
    states_mpc, actions_mpc, rewards_mpc = [], [], []
    
    for i in range(num_hours):
        action, _ = mpc_agent.predict(env_mpc)
        obs, reward, _, _, _ = env_mpc.step(action)
        states_mpc.append(obs.copy())
        actions_mpc.append(action.copy())
        rewards_mpc.append(reward)
    
    states_mpc = np.array(states_mpc)
    actions_mpc = np.array(actions_mpc)
    rewards_mpc = np.array(rewards_mpc)
    
    print(f"✅ MPC Evaluation abgeschlossen (Total Reward: {np.sum(rewards_mpc):.2f})\n")
    
    # ========== Visualisierung ==========
    print("📈 Erstelle Visualisierungen...")
    
    # Einzelne Plots
    plot_temperature_profiles(states_sac, None, agent_name="SAC", save_path=save_path)
    plot_energy_consumption(actions_sac, agent_name="SAC", save_path=save_path)
    calculate_statistics(states_sac, actions_sac, agent_name="SAC")
    
    plot_temperature_profiles(states_mpc, None, agent_name="MPC", save_path=save_path)
    plot_energy_consumption(actions_mpc, agent_name="MPC", save_path=save_path)
    calculate_statistics(states_mpc, actions_mpc, agent_name="MPC")
    
    # Vergleich
    results = {
        "SAC": (states_sac, actions_sac, rewards_sac),
        "MPC": (states_mpc, actions_mpc, rewards_mpc)
    }
    compare_agents(results, save_path=save_path)
    
    print("\n✅ Alle Ergebnisse gespeichert in: results/")


if __name__ == "__main__":
    main()