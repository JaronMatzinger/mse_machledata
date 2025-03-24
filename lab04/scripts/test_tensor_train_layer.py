import torch
import torch.nn as nn
import pytest

class TensorTrainLayer(nn.Module):
    def __init__(self, in_modes, out_modes, ranks):
        super(TensorTrainLayer, self).__init__()
        assert len(in_modes) == len(out_modes) == len(ranks) + 1
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.ranks = ranks
        self.weights = nn.ParameterList()

        # Init tensor train cores
        for i in range(len(ranks)):
            self.weights.append(nn.Parameter(torch.randn(in_modes[i], out_modes[i], ranks[i], ranks[i+1])))

    def forward(self, x):
        # Reshape input tensor to tensor train format
        tensor_train = x.view(self.in_modes[0], -1, self.out_modes[-1])

        for i in range(len(self.weights)):
            # Apply tensor contraction
            tensor_train = torch.einsum('mnr, nrm -> nm', tensor_train, self.weights[i])
        return tensor_train
    
class TensorTrainLayerTest:
    @staticmethod
    def generate_random_test_cases(num_cases=5):
        test_cases = []
        for i in range(num_cases):
            rank = torch.randint(1, 5, (1,)).item() # Random rank between 1 and 4
            num_factors = torch.randint(2, 5, (1,)).item() 

            in_modes = torch.randint(2, 6, (num_factors,)).tolist()
            out_modes = torch.randint(2, 6, (num_factors,)).tolist()
            ranks = [rank] * (num_factors - 1)

            test_cases.append((in_modes, out_modes, ranks))
        return test_cases
    
    @pytest.mark.parametrize('in_modes, out_modes, ranks', generate_random_test_cases())
    def test_tensor_train_layer(self, in_modes, out_modes, ranks):
        batch_size = torch.randint(1, 10, (1,)).item()
        input_tensor = torch.randn(batch_size, sum(in_modes))

        model = TensorTrainLayer(in_modes, out_modes, ranks)

        try: 
            output = model(input_tensor)
            assert output is not None
            print(f'Test passed for in_modes={in_modes}, out_modes={out_modes}, ranks={ranks}')
        except Exception as e:
            print(f'Test failed for in_modes={in_modes}, out_modes={out_modes}, ranks={ranks} with error: {e}')
            assert False