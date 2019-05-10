import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.feature0 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 48*160

            nn.Conv1d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128*80

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 1*256*40
        )

        self.feature1 = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 48*160

            nn.Conv1d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128*80

            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 1*256*40
        )

        self.classify = nn.Sequential(
            nn.Conv1d(512 * 2, 24, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2),  # N*24*20

            nn.Conv1d(24, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2)  # N*8*10
        )

        self.fc = nn.Sequential(
            nn.Linear(80, 1),
            nn.Sigmoid())

    def forward_once(self, x):
        x0 = self.feature0(x)
        x1 = self.feature1(x)
        return torch.cat((x0, x1), 1)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat((output1, output2), 1)

        output = self.classify(output)
        output = output.view(-1, 1, 80).contiguous()

        output = self.fc(output)
        return output
