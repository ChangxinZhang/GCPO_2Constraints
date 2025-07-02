# Game-theoretic Constrained Policy Optimization (GCPO) for Safe Reinforcement Learning
GCPO (Game-theoretic Constrained Policy Optimization) is a constrained reinforcement learning method. GCPO formulates the  CRL  problem as a task-constraints Markov game (TCMG), in which a task player strives to maximize the cumulative task rewards, while constraint players focus on reducing the constraint costs until the constraints are satisfied.


## 🔧 Project Origin

This repository is developed **based on** the following projects from [PKU-Alignment](https://github.com/PKU-Alignment):

- [Safe-Policy-Optimization (SafePO)](https://github.com/PKU-Alignment/Safe-Policy-Optimization)  
  A unified safe reinforcement learning benchmark providing baseline algorithms and evaluation tools for Safe RL.

- [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)  
  A collection of safe control environments including Safe Navigation, Safe Velocity, and Isaac Gym tasks.

---

## 🧪 Our Extension: Two-Constraint Variant

We extend the `ShadowHandOverSafeFinger` task by formalizing **two separate joint-angle constraints** as independent cost functions:

- **Constraint I**: $c_I = 1$  if $ang_2 ∉ [22.5°, 67.5°]$, else $0$   
- **Constraint II**: $c_{II} = 1$ if $ang_3 ∉ [22.5°, 67.5°]$, else $0$

where `ang_2` and `ang_3` denote the joint angles of the second and third joints of each hand's forefinger.

We evaluate multiple safe RL baselines — including **GCPO**, **Lagrangian methods**, and **CPO** — under this two-constraint setting.

> 📁 **Note**: To enable this variant of environment, move  
> `env_2constraints/ShadowHandOver_Safe_finger.py`  
> to  
> `.../safety_gymnasium/tasks/safe_isaac_gym/envs/tasks/`
