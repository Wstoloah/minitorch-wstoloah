from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List


class Module:
    """Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes
    ----------
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__[
            "_modules"
        ]  # self.__dict__ is used to access the attributes of the Module instance (not necessary in this case)
        return list(m.values())

    def train(self) -> None:
        """Set the mode of this module and all descendant modules to `train` (training mode)."""
        self.training = True  # Apply to this module
        stack = list(self._modules.values())  # Start with direct submodules
        while stack:
            child = stack.pop()
            child.training = True
            stack.extend(child._modules.values())

    def eval(self) -> None:
        """Set the mode of this module and all descendant modules to `eval` (evaluation mode)."""
        self.training = False  # Apply to this module
        stack = list(self._modules.values())  # Start with direct submodules
        while stack:
            child = stack.pop()
            child.training = False
            stack.extend(child._modules.values())

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """
        params: List[Tuple[str, Parameter]] = []

        # Include parameters of this module
        for name, param in self._parameters.items():
            params.append((name, param))

        # Stack for DFS traversal of children
        stack: List[Tuple[str, Module]] = list(self._modules.items())

        while stack:
            prefix, module = stack.pop()
            for name, param in module._parameters.items():
                full_name = f"{prefix}.{name}"
                params.append((full_name, param))
            for child_name, child_module in module._modules.items():
                stack.append((f"{prefix}.{child_name}", child_module))

        return params

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        params: List[Parameter] = list(self._parameters.values())

        # Stack for DFS traversal of child modules
        stack: List[Module] = list(self._modules.values())

        while stack:
            child = stack.pop()
            params.extend(child._parameters.values())
            stack.extend(child._modules.values())

        return params

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forwards the call to the `forward` method with the given arguments."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
