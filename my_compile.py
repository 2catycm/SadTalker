# %%
# pip install onnxconverter_common onnx onnxruntime-tools
# pip install onnxruntime-gpu
# gpt环境
# %%
import asyncio
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx



import os
os.add_dll_directory(r'D:\ProgramFiles\NVIDIA\cuda\lib\x64')
import onnxruntime as ort

# https://zhuanlan.zhihu.com/p/422290231#:~:text=%E5%9C%A8pytorch%E4%B8%AD%E8%BD%AC%E6%8D%A2%E4%B8%BAonnx%E7%9A%84%E6%A8%A1%E5%9E%8B%EF%BC%8C%E5%B0%B1%E4%B8%80%E8%A1%8C%E4%BB%A3%E7%A0%81%EF%BC%9A%20torch.onnx.export%28model%2C%20args%2C%20f%2C%20export_params%3DTrue%2C%20verbose%3DFalse%2C%20input_names%3DNone%2C,output_names%3DNone%2Cdo_constant_folding%3DTrue%2Cdynamic_axes%3DNone%2Copset_version%3D9%29%20%E5%B8%B8%E7%94%A8%E5%8F%82%E6%95%B0%EF%BC%9A%201.model%3Atorch.nn.model%20%E8%A6%81%E5%AF%BC%E5%87%BA%E7%9A%84%E6%A8%A1%E5%9E%8B%202.args%3Atuple%20or%20tensor%20%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E5%8F%82%E6%95%B0%E3%80%82

import tempfile
import inspect


class TorchCompile(nn.Module):
    def __init__(
        self, model: nn.Module, file_path=None, exist_recompile=True, verbose=True
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.session = None

        if file_path is None:
            file_path = tempfile.NamedTemporaryFile(
                prefix="MyTorchCompile", suffix=".onnx"
            ).name
        elif not exist_recompile and os.path.isfile(file_path):
            try:
                self.session = ort.InferenceSession(
                    file_path, providers=["CUDAExecutionProvider"]
                )
            except Exception as e:
                print(
                    f'Cannot load model from {file_path} due to "{e}", will recompile later.'
                )
                self.session = None
        self.model = model
        self.file_path = file_path

    def init_session_from_model(self, input_tensor):
        if self.verbose:
            print(f"First time calling function {self.model}, compiling for you!")
        # arg_size = tuple(input_tensor.size())

        # print(f"Input size infered to be {arg_size}")
        torch.onnx.export(
            self.model,
            #   arg_size,
            input_tensor,
            self.file_path,
            export_params=True,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            # dynamic_axes=None,
            dynamic_axes={
                "input": {
                    0: "batch",
                },
                "output": {0: "batch"},
            },
            # opset_version=9
            opset_version=11,
        )
        from onnxruntime_tools import optimizer
        import onnx

        # https://zhuanlan.zhihu.com/p/459875044
        # https://github.com/kenwaytis/faster-SadTalker-API/commit/3bd28ea49e2a48c4b46d9bfe8b693f660fd38cf2
        try:
            # original_model = onnx.load(self.file_path)
            # passes = ['fuse_bn_into_conv']
            # onnx.save(optimized_model, self.file_path)

            optimized_model = optimizer.optimize_model(
                self.file_path,
                # self.file_path = optimizer.optimize_by_onnxruntime(self.file_path,
                # optimized_model_path=self.file_path,
                opt_level=99,
                use_gpu=True,
            )
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file(self.file_path)
            # onnx.save(optimized_model, self.file_path)

        except Exception as e:
            # raise e
            print(f'due to "{e}", we skip onnx optimize.')

        self.session = ort.InferenceSession(
            self.file_path, providers=["TensorrtExecutionProvider", 
                                       "CUDAExecutionProvider"]
        )
        print(self.session.get_providers())
        if self.verbose:
            for var in self.session.get_inputs():
                print(
                    f"Input var '{var.name}' has shape={var.shape} and type={var.type} ."
                )
            for var in self.session.get_outputs():
                print(
                    f"Output var '{var.name}' has shape={var.shape} and type={var.type} ."
                )
            print(f"Compiled {self.model} to file '{self.file_path}' !")

    async def forward_async(self, input_tensor):
        if self.session is None:
            self.init_session_from_model(input_tensor)
        # return model(input_tensor)
        await_results = None
        def callback(results: np.ndarray, user_data, err: str) -> None:
            nonlocal await_results
            await_results = (results, user_data, err)
        self.session.run_async(
            ["output"],
            {
                # "input"
                self.session.get_inputs()[0]
                .name: input_tensor.detach()
                .cpu()
                .numpy()
            },
            callback=callback,
            user_data=None
        )
        while await_results is None:
            await asyncio.sleep(0.01)
        return torch.tensor(await_results[0][0], device=input_tensor.device)
    
    def forward(self, input_tensor):
        if self.session is None:
            self.init_session_from_model(input_tensor)
        # return model(input_tensor)
        outputs = self.session.run(
            ["output"],

            {
                # "input"
                self.session.get_inputs()[0]
                .name: input_tensor.detach()
                .cpu()
                .numpy()
            },
        )
        return torch.tensor(outputs[0], device=input_tensor.device)


#%%
def test_compile():
    class Add(nn.Module):
        def __init__(self):
            super(Add, self).__init__()
            self.weight = nn.Parameter(torch.rand(1).float()).float()

        def forward(self, x: torch.FloatTensor):
            return x + self.weight


    model = Add()
    x = torch.rand((1, 2)).float()
    res = model(x)
    print(res)
    model = TorchCompile(model, "test.onnx")
    # model = TorchCompile(model)
    res2 = model(x)
    assert torch.allclose(res, res2, rtol=1e-3, atol=1e-3)
    # x = torch.Tensor([1,2,3])
    x = torch.rand((3, 2)).cuda()
    res3 = model(x)
    assert x.device == res3.device

import pytest    
@pytest.mark.asyncio
async def test_async():
    class Add(nn.Module):
        def __init__(self):
            super(Add, self).__init__()
            self.weight = nn.Parameter(torch.rand(1).float()).float()

        def forward(self, x: torch.FloatTensor):
            return x + self.weight
    model = Add()
    model = TorchCompile(model, "test.onnx")
    x = torch.rand((1, 2)).float()
    res3 = model(x)
    res4 = await model.forward_async(x)
    assert torch.allclose(res3, res4, rtol=1e-3, atol=1e-3)
    print(res4)


def test_many_args():
    class Add(nn.Module):
        def __init__(self):
            super(Add, self).__init__()
            self.weight = nn.Parameter(torch.rand(1).float()).float()

        def forward(self, x: torch.FloatTensor, y: torch.FloatTensor, *args, **kwargs):
            return x + y + self.weight
    sig = inspect.signature(Add.forward)
    params = sig.parameters

    for name, param in params.items():
        print(f"Parameter name: {name}")
        print(f"Default value: {param.default if param.default is not param.empty else 'No default value'}")
        print(f"Annotation: {param.annotation if param.annotation is not param.empty else 'No annotation'}")