import os

import re
from setuptools import setup
from pkg_resources import parse_version
import subprocess
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR

def min_supported_compute_capability(cuda_version):
	if cuda_version >= parse_version("12.0"):
		return 50
	else:
		return 20

def max_supported_compute_capability(cuda_version):
	if cuda_version < parse_version("11.0"):
		return 75
	elif cuda_version < parse_version("11.1"):
		return 80
	elif cuda_version < parse_version("11.8"):
		return 86
	else:
		return 90

# Find version of permutohedral_encoding by scraping CMakeLists.txt
# with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
# 	for line in cmakelists.readlines():
# 		if line.strip().startswith("VERSION"):
# 			VERSION = line.split("VERSION")[-1].strip()
# 			break
VERSION=1.0

print(f"Building PyTorch extension for permutohedral_encoding version {VERSION}")

ext_modules = []

if "PENC_CUDA_ARCHITECTURES" in os.environ and os.environ["PENC_CUDA_ARCHITECTURES"]:
	compute_capabilities = [int(x) for x in os.environ["PENC_CUDA_ARCHITECTURES"].replace(";", ",").split(",")]
	print(f"Obtained compute capabilities {compute_capabilities} from environment variable PENC_CUDA_ARCHITECTURES")
elif torch.cuda.is_available():
	major, minor = torch.cuda.get_device_capability()
	compute_capabilities = [major * 10 + minor]
	print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
	raise EnvironmentError("Unknown compute capability. Specify the target compute capabilities in the PENC_CUDA_ARCHITECTURES environment variable or install PyTorch with the CUDA backend to detect it automatically.")


if os.name == "nt":
	def find_cl_path():
		import glob
		for executable in ["Program Files (x86)", "Program Files"]:
			for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
				paths = sorted(glob.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)
				if paths:
					return paths[0]

	# If cl.exe is not on path, try to find it.
	if os.system("where cl.exe >nul 2>nul") != 0:
		cl_path = find_cl_path()
		if cl_path is None:
			raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
		os.environ["PATH"] += ";" + cl_path
	else:
		# cl.exe was found in PATH, so we can assume that the user is already in a developer command prompt
		# In this case, BuildExtensions requires the following environment variable to be set such that it
		# won't try to activate a developer command prompt a second time.
		os.environ["DISTUTILS_USE_SDK"] = "1"

# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
	nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
	cuda_version = re.search(r"release (\S+),", nvcc_out)

	if cuda_version:
		cuda_version = parse_version(cuda_version.group(1))
		print(f"Detected CUDA version {cuda_version}")
		supported_compute_capabilities = [
			cc for cc in compute_capabilities if cc >= min_supported_compute_capability(cuda_version) and cc <= max_supported_compute_capability(cuda_version)
		]

		if not supported_compute_capabilities:
			supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

		if supported_compute_capabilities != compute_capabilities:
			print(f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead.")
			compute_capabilities = supported_compute_capabilities

min_compute_capability = min(compute_capabilities)

base_nvcc_flags = [
	"-std=c++14",
	"--extended-lambda",
	"--expt-relaxed-constexpr",
	# The following definitions must be undefined
	# since permutohedral_encoding requires half-precision operation.
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
	base_cflags = ["-std=c++14"]
	base_nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
	base_cflags = ["/std:c++14"]


# Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# List of sources.
bindings_dir = os.path.dirname(__file__)
# root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))
root_dir = os.path.abspath(bindings_dir)
base_definitions = []
base_source_files = [
	"./src/Encoding.cu",
	"./src/PyBridge.cxx",
]



def make_extension(compute_capability):
	nvcc_flags = base_nvcc_flags + [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]
	definitions = base_definitions + [f"-DPENC_MIN_GPU_ARCH={compute_capability}"]

	source_files = base_source_files 

	nvcc_flags = nvcc_flags + definitions
	cflags = base_cflags + definitions

	ext = CUDAExtension(
		# name=f"permutohedral_encoding._{compute_capability}_C",
		name=f"permutohedral_encoding_bindings._{compute_capability}_C",
		# name=f"permutohedral_encoding",
		sources=source_files,
		include_dirs=[
			"%s/include" % root_dir,
			"%s/kernels" % root_dir,
			"%s/deps" % root_dir,
		],
		extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
		libraries=["cuda", "cudadevrt", "cudart_static"],
	)
	return ext

# ext_modules = [make_extension(comp) for comp in compute_capabilities]
print("Building for compute capability: ", min_compute_capability)
ext_modules = [make_extension(min_compute_capability)]
# print("-------------------ext_modules",ext_modules)

setup(
	name="permutohedral_encoding",
	version=VERSION,
	description="permutohedral_encoding extension for PyTorch",
	long_description="permutohedral_encoding extension for PyTorch",
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: GPU :: NVIDIA CUDA",
		"License :: BSD 3-Clause",
		"Programming Language :: C++",
		"Programming Language :: CUDA",
		"Programming Language :: Python :: 3 :: Only",
		"Topic :: Multimedia :: Graphics",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Image Processing",
	],
	keywords="PyTorch,machine learning",
	url="https://github.com/RaduAlexandru/permutohedral_encoding",
	author="Radu Alexandru Rosu",
	author_email="rosu@ais.uni-bonn.de",
	maintainer="Radu Alexandru Rosu",
	maintainer_email="rosu@ais.uni-bonn.de",
	download_url=f"https://github.com/RaduAlexandru/permutohedral_encoding",
	license="BSD 3-Clause \"New\" or \"Revised\" License",
	packages=["permutohedral_encoding"],
	package_dir={"permutohedral_encoding": "src"},
	package_data={"permutohedral_encoding": ["pytorch_modules/*"]},
	install_requires=[],
	# include_package_data=True,
	zip_safe=False,
	ext_modules=ext_modules,
	cmdclass={"build_ext": BuildExtension}
)
