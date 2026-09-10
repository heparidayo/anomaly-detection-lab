"""입력을 작게 압축한 뒤 원래 크기로 복원하는 교육용 CNN AutoEncoder."""

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        # 아래 shape는 128×128 입력 기준. B(배치 크기)는 모든 층에서 같다.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # [B,16,64,64]
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),  # [B,32,32,32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B,64,16,16]
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, 4, 2, 1),  # [B,64,8,8] (기본값)
            nn.ReLU(),
        )
        # 압축된 표현만으로 복원하게 하여 정상 부품의 반복되는 패턴을 배우게 한다.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1),  # [B,64,16,16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B,32,32,32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # [B,16,64,64]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # [B,3,128,128]
            nn.Sigmoid(),  # 입력과 같은 0~1 범위의 RGB 값을 만든다.
        )

    def forward(self, images):
        latent = self.encoder(images)
        reconstructed = self.decoder(latent)
        return reconstructed
