from lean_dojo import *

repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "fd14c4c8b29cc74a082e5ae6f64c2fb25b28e15e")
theorem = Theorem(repo, "Lean4Example.lean", "hello_world")

with Dojo(theorem) as (dojo, init_state):
  print(init_state)
  result = dojo.run_tac(init_state, "rw [add_assoc, add_comm b, ←add_assoc]")
  assert isinstance(result, ProofFinished)
  print(result)
