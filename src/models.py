import torch

def _concat_orography(x: torch.Tensor, orography: torch.Tensor) -> torch.Tensor:

    if orography is None:
        raise ValueError('orography tensor is required')

    if orography.device != x.device:
        orography = orography.to(x.device)

    if orography.dim() == 1:
        orography = orography.unsqueeze(0)

    if orography.dim() == 2 and orography.shape[0] != x.shape[0]:
        orography = orography.unsqueeze(0)

    if orography.shape[0] != x.shape[0]:
        if orography.shape[0] == 1:
            orography = orography.expand(x.shape[0], *orography.shape[1:])
        else:
            raise ValueError('orography batch size does not match x batch size')

    orography = torch.flatten(orography, start_dim=1)
    x = torch.cat((x, orography), dim=1)

    return x

class DeepESDtas(torch.nn.Module):

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool,
                 orography: torch.Tensor=None):

        super(DeepESDtas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic
        self.register_buffer('orography', orography)
        out_in_features = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            if self.orography is not None:
                self.out_mean = torch.nn.LazyLinear(out_features=self.y_shape[1])
                self.out_log_var = torch.nn.LazyLinear(out_features=self.y_shape[1])
            else:
                self.out_mean = torch.nn.Linear(in_features=out_in_features,
                                                out_features=self.y_shape[1])

                self.out_log_var = torch.nn.Linear(in_features=out_in_features,
                                                   out_features=self.y_shape[1])

        else:
            if self.orography is not None:
                self.out = torch.nn.LazyLinear(out_features=self.y_shape[1])
            else:
                self.out = torch.nn.Linear(in_features=out_in_features,
                                           out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)
        if self.orography is not None:
            x = _concat_orography(x=x, orography=self.orography)

        if self.stochastic:
            mean = self.out_mean(x)
            log_var = self.out_log_var(x)
            out = torch.cat((mean, log_var), dim=1)
        else:
            out = self.out(x)

        return out

class DeepESDpr(torch.nn.Module):

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool,
                 last_relu: bool=False, orography: torch.Tensor=None):

        super(DeepESDpr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic
        self.last_relu = last_relu
        self.register_buffer('orography', orography)
        out_in_features = self.x_shape[2] * self.x_shape[3] * self.filters_last_conv

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            if self.orography is not None:
                self.p = torch.nn.LazyLinear(out_features=self.y_shape[1])
                self.log_shape = torch.nn.LazyLinear(out_features=self.y_shape[1])
                self.log_scale = torch.nn.LazyLinear(out_features=self.y_shape[1])
            else:
                self.p = torch.nn.Linear(in_features=out_in_features,
                                         out_features=self.y_shape[1])

                self.log_shape = torch.nn.Linear(in_features=out_in_features,
                                                 out_features=self.y_shape[1])

                self.log_scale = torch.nn.Linear(in_features=out_in_features,
                                                 out_features=self.y_shape[1])

        else:
            if self.orography is not None:
                self.out = torch.nn.LazyLinear(out_features=self.y_shape[1])
            else:
                self.out = torch.nn.Linear(in_features=out_in_features,
                                           out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)
        if self.orography is not None:
            x = _concat_orography(x=x, orography=self.orography)

        if self.stochastic:
            p = self.p(x)
            p = torch.sigmoid(p)

            log_shape = self.log_shape(x)
            log_scale = self.log_scale(x)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x)
            if self.last_relu: out = torch.relu(out)

        return out

