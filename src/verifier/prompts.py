LLEMMA_4S_PROMPT = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

Tactic state:
---
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
---
Next tactic:
---
rintro s t ⟨u, a, hr, he⟩
---

Tactic state:
---
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
---
Next tactic:
---
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
---

Tactic state:
---
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
---
Next tactic:
---
rw [← h.gcd_eq_one]
---

Tactic state:
---
{state}
---
Next tactic:
---""" 

def _llemma_prompt_fewshot(ts):
    return LLEMMA_4S_PROMPT.format(state=ts)

INSTRUCTION_TEMPLATE = """Given the Lean 4 tactic state, suggest a next tactic.

Tactic state:
---
{state}
---
Next tactic:
---
{target}
"""

INSTRUCTION_PROMPT_TEMPLATE = """Given the Lean 4 tactic state, suggest a next tactic.

Tactic state:
---
{state}
---
Next tactic:
---
"""