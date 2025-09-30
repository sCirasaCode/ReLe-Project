from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Controller.MPC_Controller import MPCAgent
from BEAR.Utils.utils_building import ParameterGenerator
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env

def plot_temperature_profiles(states, zone_names, hours=24):
    """ Visualisiert Temperaturprofile für Schlüsselzonen über einen bestimmten Zeitraum. """
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
        ax.plot(range(hours), temperatures, label=name)
    
    ax.set_xlabel('Stunden')
    ax.set_ylabel('Temperatur (°C)')
    ax.set_title('Temperaturprofile der kritischen Zonen über 24 Stunden')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("temperature_profiles.png")

def plot_energy_consumption(actions, hours=24):
    """ Visualisiert den Energieverbrauch über einen bestimmten Zeitraum. """
    total_energy = []
    for i in range(hours):
        """ Berechnung des Energieverbrauchs als Summe der absoluten Aktionswerte """
        energy = sum(abs(action) for action in actions[i])
        total_energy.append(energy)
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(hours), total_energy, marker='o', linewidth=2)
    plt.xlabel('Stunden')
    plt.ylabel('Energieverbrauch (kWh)')
    plt.title('Energieverbrauch über 24 Stunden')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("energy_consumption.png")

def calculate_statistics(states, actions):
    """ Berechnet und druckt statistische Kennzahlen für Temperaturen und Energieverbrauch. """
    temperatures = np.array(states)
    actions = np.array(actions)

    temp_mean = np.mean(temperatures, axis=0)
    temp_std = np.std(temperatures, axis=0)
    energy_mean = np.mean(np.sum(np.abs(actions), axis=1))
    energy_std = np.std(np.sum(np.abs(actions), axis=1))

    print("Temperaturstatistiken (Mittelwert ± Standardabweichung):")
    for i, (mean, std) in enumerate(zip(temp_mean, temp_std)):
        print(f"Zone {i}: {mean:.2f}°C ± {std:.2f}°C")
    
    print(f"\nEnergieverbrauch (Mittelwert ± Standardabweichung): {energy_mean:.2f} kWh ± {energy_std:.2f} kWh")    

# --------------------
# Main 
# --------------------
def main():
    num_hours = 24
    Parameter = ParameterGenerator('Hospital', 'Hot_Humid', 'Denver')
    env = BuildingEnvReal(Parameter)
    env.zone_names = env.get_zone_names()
    env.reset()

    """ MPC Agent initialisieren """
    mpc_agent = MPCAgent(env, gamma=env.gamma, safety_margin=0.96, planning_steps=10)
    states = [] # Liste zur Speicherung der Zustände
    actions = [] # Liste zur Speicherung der Aktionen

    for i in range(num_hours):
        action, _ = mpc_agent.predict(env)
        obs, _, _, _, _ = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())

    """ Ergebnisse visualisieren """
    plot_temperature_profiles(states, env.zone_names)
    plot_energy_consumption(actions)

    """ Statistiken berechnen und printen """
    calculate_statistics(states, actions)

if __name__ == "__main__":
    main()