from dqn_eval import evaluate
from Quantum_Policy import make_env,QNetwork
import torch
import time
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model_path = "/home/zecplusplus/Documents/Fraunhofer/Anki-Overdrive-Code/Anki-Overdrive-Reinforcement-Learning/runs/Anki_Overdrive_Quantum__Quantum_Policy__1__1722945587/Quantum_Policy.cleanrl_model"
    env_id= "Anki_Overdrive_Quantum"
    run_name = f"Anki_Overdrive_Quantum__Quantum_Policy__evaluate__{int(time.time())}"
    episodic_returns = evaluate(
        model_path,
        make_env,
        env_id,
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=QNetwork,
        device=device,
        epsilon=0.00,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)
