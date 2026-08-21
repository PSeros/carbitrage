# Method

## Two sides of one assumption

**What is not monetised.** Range, charging time, comfort, noise, the nuisance of
a dead car on a Monday morning — none of it. The library quantifies what is
quantifiable and stays silent on the rest. Conflating the two is exactly the
error that makes spreadsheet TCO comparisons untrustworthy, so where a
difference cannot be defended in euros it does not appear.

**What licenses deciding on cost alone.** That silence is only acceptable
because of an assumption doing a great deal of work: every alternative is taken
to deliver **the same service over the same horizon**. That assumption is
fragile. A larger vehicle, or one with materially different range, is *not* the
same service — and in that case the comparison is **invalid rather than merely
incomplete**. No amount of methodological care downstream repairs a comparison
between things that are not comparable.

These are the same point seen from opposite sides. Check the premise before you
read the number.

Note also that an absence of *operating revenue* is not an absence of inflows.
Grants, recurring credits, disposal proceeds of the incumbent at t=0 and
terminal residual value are all genuine inflows; they are netted inside the
stream and never dropped. What is absent is revenue *attributable to the asset*,
because the vehicle earns nothing.

## Terminology

In German this is a **Kapitalwertvergleich** or **Barwertminimierung**.
*Kostenvergleichsrechnung* denotes the **static, undiscounted** method in German
cost accounting and is the wrong label for what this library does.

## Method note

The alternatives here are **not identical repetitions**: the same car bought
today and bought in two years differ in price vintage, in subsidy availability,
and in *age at the horizon* — six years versus four. So:

- EAC-as-infinite-chain is **not** licensed as the primary method. The
  defensible one is the explicit finite common horizon with age-consistent
  terminal values, which is what `ReplacementChain` builds.
- Consequently **the period at which a leg starts is an economic quantity**, not
  plumbing: it fixes the asset's age and hence its residual, the vintage of its
  price, and whether and when a subsidy disburses.

The "keep it or replace it" question is the classical defender–challenger
problem (Terborgh 1949; Grant/Ireson/Leavenworth), and optimal replacement age
traces to Hotelling (1925) and Preinreich (1940). Deferring under uncertainty is
a real option (McDonald & Siegel 1986; Dixit & Pindyck 1994) — deterministic NPV
systematically *understates* the value of waiting, and this library does not
price that flexibility.

None of these are "proven" the way a theorem is. They are normative decision
rules resting on stated assumptions, chiefly a capital market where you can lend
and borrow at the discount rate. The genuinely uncertain part of any result here
is not the method but the **residual value curve**, and no methodological rigour
repairs a wrong depreciation assumption.

### Real or nominal residuals

A depreciation rate read off observed used-asset prices is a **nominal** rate of
price decline — it already contains inflation. Applied to today's nominal price
it yields a future nominal price, correctly discounted at a nominal rate. On a
real basis, restate the rate as `1 - (1 - nominal) / (1 + inflation)`. The
library cannot tell which one you handed it, so this is documented rather than
guessed at.

