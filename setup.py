from setuptools import Extension, setup

# ctypes keeps the search implementation independent of Python and neural-runtime headers.
setup(
    ext_modules=[
        Extension(
            "issen_ccg.adapters._search",
            ["src/issen_ccg/adapters/search.cpp"],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3", "-DNDEBUG"],
        )
    ]
)
