
import torch
from configs.dreamerv2_config import DreamerV2Config
from agents.dreamerv2_agent import DreamerV2Agent

def test_device_init():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on {device}")
    else:
        print("CUDA not available, skipping device mismatch test (but checking CPU behavior)")
        device = torch.device("cpu")

    config = DreamerV2Config()
    # Mock obs shape and action size
    obs_shape = (4, 3, 64, 64)
    n_actions = 4
    
    agent = DreamerV2Agent(obs_shape, n_actions, config).to(device)
    
    print("Agent moved to device.")
    
    # Test 1: Explicit reset (already fixed in agent code, but good to check)
    agent.reset()
    print("Agent reset called.")
    assert agent.prev_rssm_state.stoch.device.type == device.type
    print("Reset tensors on correct device.")

    # Test 2: Implicit call via _init_rssm_state without kwargs
    # This simulates what happens in representation_loss
    rssm_state = agent.rssm._init_rssm_state(1)
    print(f"Implicit _init_rssm_state device: {rssm_state.stoch.device}")
    assert rssm_state.stoch.device.type == device.type
    print("Implicit init tensors on correct device!")

if __name__ == "__main__":
    test_device_init()
