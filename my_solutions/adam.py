import torch

class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        # Convert generator/list of params into a concrete list
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        # Initialize moment vectors
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0  # Timestep counter for bias correction

    def step(self):
        self.t += 1
        b1, b2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad

            # 1. Update biased 1st and 2nd moment estimate (in-place)
            self.m[i] = b1 * self.m[i] + (1 - b1) * grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * (grad ** 2)

            # 2. Compute bias-corrected estimates using timestep t
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)

            with torch.no_grad():
                # If I would not apply this, gradie
                # 3. Update parameters in-place
                p-=self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                # PyTorch standard practice is detaching and zeroing or setting to None
                with torch.no_grad():
                    p.grad = torch.zeros_like(p)