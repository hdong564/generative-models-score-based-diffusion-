# Noise Conditioned Score Network

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# 1. Swiss Roll 데이터셋 생성
def get_swiss_roll_data(n_samples=10000):
    data, _ = make_swiss_roll(n_samples=n_samples, noise=0.3)
    data = data[:, [0, 2]]  # 2D: x and z
    data = data / 10.0      # normalize
    return torch.tensor(data, dtype=torch.float32)

# 2. ScoreNet 정의
class ScoreNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, input_dim)

    def forward(self, x, sigma):
        h = torch.cat([x, sigma], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

# 3. 학습 함수
def train_score_model(model, data, sigmas, steps=3000, batch_size=128, lr=0.001, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        indices = torch.randint(0, data.shape[0], (batch_size,)) # data.shape[0] = 10000
        x0 = data[indices].to(device)
        # print(data[indices])

        sigma_idx = torch.randint(0, len(sigmas), (batch_size,), device=device)
        sigma = sigmas[sigma_idx].unsqueeze(1)
        
        noise = torch.randn_like(x0)
        
        x_tilde = x0 + sigma * noise
        target = -noise / sigma

        score_pred = model(x_tilde, sigma)
        loss = ((score_pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

# 4. 샘플링 함수
@torch.no_grad()
def sample(score_model, sigmas, num_samples=1000, n_steps_each=10, step_lr=0.001, device='cpu'):
    score_model.eval()
    input_dim = 2
    x = torch.randn(num_samples, input_dim).to(device) * sigmas[-1]
    for sigma in reversed(sigmas):
        sigma_tensor = sigma.expand(num_samples, 1)
        for _ in range(n_steps_each):
            grad = score_model(x, sigma_tensor)
            noise = torch.randn_like(x)
            x = x + step_lr * grad + torch.sqrt(torch.tensor(2.0 * step_lr)).to(device) * noise
    return x.cpu().numpy()

# 5. 시각화 함수 (원본 + 샘플)
def visualize_comparison(original, samples, title='Original vs Generated'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(original[:, 0], original[:, 1], s=5, alpha=0.6, c='dodgerblue')
    axs[0].set_title('Original Swiss Roll Data')
    axs[0].axis('equal')
    axs[0].grid(True)

    axs[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6, c='crimson')
    axs[1].set_title('Generated Samples')
    axs[1].axis('equal')
    axs[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 6. 메인 실행
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 2

    # noise schedule (log spaced)
    sigmas = torch.tensor(np.exp(np.linspace(np.log(0.01), np.log(1.0), 10)), dtype=torch.float32).to(device)

    # 데이터 준비
    data = get_swiss_roll_data(n_samples=10000)
    score_model = ScoreNet(input_dim).to(device)

    # 학습
    train_score_model(score_model, data, sigmas, steps=1000000, device=device)

    # 샘플 생성
    samples = sample(score_model, sigmas, num_samples=1000, n_steps_each=100, device=device)

    # 시각화 (원본 + 생성)
    visualize_comparison(data.numpy(), samples)