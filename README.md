# code on spectral methods for managing SESs


# next steps


**!!** = for next week

## writing

We're just now gettting to some of the questions: asymetric info, total damage/benefits, et

**!!** Need document/pres for group

- **!!** consolidate writeup / scenarios to single, clear writeup of scenarios
- **!!** visual of the scenarios / groups: info, control (conceptual fig: contorl and info as different directed edges; coming from a _group of nodes_)
- visual related to total damage
- existing visual of 

**!!** need asks for group work on Friday call:

- motivation/references/paragraphs
- lanaguage/framing/terms

## code

for each patch compute:

$patch level p_i \approx \gamma * \lambda / \beta u$ with  $u$ right eigen vector of A

occupancy probability near threshold. This gives us a way to look at expected total "damage"  \gamma * \lambda /\beta \sum_{i \in Manager} u_i. We can compare this for globally-knowledgable/concerned manager vs locally-knowledgable/concerned. Then can show tradeoff. expect in some network topos the greedy/locally focused has different effects.
expect more noticable when we combine budgets -- someone in a safe zone could contribute to the global more. 

todo for code

- **!!** work up something like this in the code
- **!!** visual related to total damage


## future

- could have different objective w/ weighted sum involving damage etc; this involves opt for eigenvector perturb not the eigen value
