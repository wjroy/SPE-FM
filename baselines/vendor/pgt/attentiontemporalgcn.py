import torch
from .temporalgcn import TGCN
from .temporalgcn import TGCN2
from torch_geometric.nn import GCNConv


class A3TGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True
    ):
        super(A3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):
            H_accum = H_accum + probs[period] * self._base_tgcn(
                X[:, :, period], edge_index, edge_weight, H
            )
        return H_accum


class A3TGCN2(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        batch_size:int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True):
        super(A3TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        H_accum = 0
        probs = torch.nn.functional.softmax(self._attention, dim=0)
        for period in range(self.periods):

            H_accum = H_accum + probs[period] * self._base_tgcn( X[:, :, :, period], edge_index, edge_weight, H)

        return H_accum
