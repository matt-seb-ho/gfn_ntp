from typing import Optional
from loguru import logger
from proof_flow.src.utils import prepare_environment_for_lean_dojo
prepare_environment_for_lean_dojo()
from lean_dojo import Dojo, TacticState, Theorem


class LeanEnvCache:
    def __init__(
        self, 
        timeout: int, 
        additional_imports: Optional[list[str]] = None
    ):
        self.cache = {}
        self.timeout = timeout
        self.additional_imports = additional_imports or []

    def get(
        self,
        theorem: Theorem,
        timeout: Optional[int] = None,
        additional_imports: Optional[list[str]] = None,
    ) -> tuple[Dojo, TacticState]:
        # return cached environment if exists
        if theorem.uid in self.cache:
            return self.cache[theorem.uid]
        # else, create a new environment 
        timeout = timeout or self.timeout
        extra_imports = additional_imports or self.additional_imports
        try:
            dojo = Dojo(
                theorem, 
                timeout=timeout, 
                additional_imports=extra_imports,
            )
            dojo, initial_state = dojo.__enter__()
            self.cache[theorem.uid] = (dojo, initial_state)
            return dojo, initial_state
        except ValueError as e:
            # LeanDojo relies on open file descriptors to communicate with Lean
            # OS only lets us open a certain number of files
            # We currently deal with this by emptying the cache to make room
            # TODO: add finer grain control over cache eviction

            # raise any other error besides the file descriptor one
            if "filedescriptor out of range" not in str(e):
                raise e
            logger.info("filedescriptor out of range, clearing cache")
            # empty cache and retry
            self.clear()
            _dojo = Dojo(
                theorem, 
                timeout=timeout, 
                additional_imports=extra_imports
            )
            dojo, initial_state = _dojo.__enter__()
            self.cache[theorem.uid] = (dojo, initial_state)
            return dojo, initial_state
    
    def clear(self) -> None:
        keys = list(self.cache.keys())
        for key in keys:
            dojo_instance, _ = self.cache.pop(key)
            dojo_instance.__exit__(None, None, None)
        logger.info("dojo cache cleared")
