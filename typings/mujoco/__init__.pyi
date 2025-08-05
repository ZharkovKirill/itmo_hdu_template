from mujoco._callbacks import *
from mujoco._constants import *
from mujoco._enums import *
from mujoco._errors import *
from mujoco._functions import *
from mujoco._render import *
from mujoco._specs import *
from mujoco._structs import *
from mujoco.gl_context import *
from _typeshed import Incomplete
from mujoco import _specs
from mujoco.renderer import Renderer as Renderer
from typing import Any, IO, Sequence
from typing_extensions import TypeAlias

proc_translated: Incomplete
is_rosetta: Incomplete
MjStruct: TypeAlias

def to_zip(spec: _specs.MjSpec, file: str | IO[bytes]) -> None: ...
def from_zip(file: str | IO[bytes]) -> _specs.MjSpec: ...

class _MjBindModel:
    elements: Incomplete
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...

class _MjBindData:
    elements: Incomplete
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...

HEADERS_DIR: Incomplete
PLUGINS_DIR: Incomplete
PLUGIN_HANDLES: Incomplete
__version__: Incomplete
