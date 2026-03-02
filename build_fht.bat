@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Projects\QwenSpinQuant\fast-hadamard-transform
set TORCH_CUDA_ARCH_LIST=8.9
set DISTUTILS_USE_SDK=1
C:\Projects\QwenSpinQuant\.venv\Scripts\python setup.py build_ext --inplace
if %ERRORLEVEL% EQU 0 (
    C:\Projects\QwenSpinQuant\.venv\Scripts\python -m pip install -e .
)
