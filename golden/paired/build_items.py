#!/usr/bin/env python3
"""Author the paired-capability prompt set, deterministically, keys and all.

# The shape of an item, and why it is one token

Every item is a four-option multiple choice question ending in `Answer:`, and
the whole measurement is the single greedy token that follows. That is not a
stylistic choice, it is what makes the experiment affordable: the BF16 reference
costs one full pass over 531.9 GB of checkpoint per FORWARD, so a first token
costs one pass for the entire set and every additional generated token would
cost another. Forty items answered in one token is one stream; forty items
answered in sixteen tokens is seventeen.

What that buys, and what it gives up, stated plainly: it measures whether the
two models pick the same answer, and it does NOT measure long-form generation,
chain-of-thought, or anything about the 2nd..nth token. A quantisation fault
that only shows up after 200 tokens of prose is invisible here. This is the
cheap end of a ladder, not the whole ladder.

# Where the items come from

Hand-written here, on 2026-08-13, for this harness. NOT drawn from MMLU, ARC,
GSM8K or any public test set — deliberately, because the one thing a public set
cannot tell you is whether the model has seen it. The prompt TEMPLATE is the
lm-eval-harness multiple-choice shape (question, lettered options, `Answer:`),
which is the only borrowed part and is not scored.

Contamination is not fully escapable for the `factual` family: those facts are
public and the model has certainly seen them, which is the point — it is
testing recall. It IS escapable for `retrieval`, whose passages are invented
here and describe entities that do not exist, so a right answer can only come
from reading the context.

# The keys are computed, not asserted

`arith` answers come from evaluating the arithmetic in python, and its
distractors from evaluating a NAMED wrong procedure (the carry dropped, the
operations in the wrong order, the final subtraction skipped) — so a plausible
distractor is a mistake with a name, not a number I made up. `code` answers come
from `exec`-ing the snippet the prompt shows. `constraint` answers come from
applying the rule the prompt states. A hand-keyed answer key is a second thing
to be wrong, and three of the five families do not have one.

`factual` and the `retrieval` needle are hand-keyed; the retrieval key is at
least mechanically checkable against the passage the same script generated.

# Open and sealed

Two files. `items_open.json` is the set that may inform a change. `holdout/
items_sealed.json` is the set that may not, ever — it exists to answer "has
tuning against the open set bought anything real?" and it stops being able to
answer that the moment anyone looks at it while tuning. It is a separate file in
a separate directory so that "I only ran the open set" is the default and
running the sealed one is a deliberate act.

The reference is batched, so the sealed set is nearly free to run at the same
time (one stream covers both). Running it is cheap; USING it is what has to be
rationed.

  build_items.py <ckpt> <outdir>
"""
import json
import os
import sys

LETTERS = "ABCD"

TEMPLATE = (
    "The following is a multiple choice question. "
    "Answer with a single letter.\n\n"
    "Question: {q}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def item(key, family, question, options, answer, source, note=None, passage=None):
    """One item, options and key still separable.

    The prompt string is NOT built here. Which letter carries the right answer
    is decided in `rebalance` once the whole family is known, because an
    unbalanced key is a free point for a model with a letter bias — and a model
    that has just been quantised is exactly the kind of thing that might
    acquire one. Four options, ten of each letter, so the bias floor and the
    chance floor are the same 25%.
    """
    assert len(options) == 4, (key, options)
    assert len(set(map(str, options))) == 4, ("duplicate option", key, options)
    assert 0 <= answer < 4, (key, answer)
    return {
        "key": key,
        "family": family,
        "question": question,
        "options": [str(x) for x in options],
        "answer": answer,
        "source": source,
        **({"passage": passage} if passage else {}),
        **({"note": note} if note else {}),
    }


def rebalance(items):
    """Rotate each item's options so the correct letter cycles A, B, C, D.

    A rotation keeps the option SET intact, so no distractor becomes more or
    less plausible; only its label moves. Within a family the items are taken
    in authoring order, which makes this deterministic and re-derivable rather
    than a seed nobody will reproduce.
    """
    seen = {}
    for it in items:
        i = seen.get(it["family"], 0)
        seen[it["family"]] = i + 1
        want = i % 4
        k = (want - it["answer"]) % 4
        if k:
            it["options"] = it["options"][-k:] + it["options"][:-k]
        it["answer"] = want
    return items


def render(it):
    """Format the prompt, and pin the letter the key now sits on."""
    o = it["options"]
    tpl = RET_TEMPLATE if "passage" in it else TEMPLATE
    kw = dict(q=it["question"], a=o[0], b=o[1], c=o[2], d=o[3])
    if "passage" in it:
        kw["p"] = it["passage"]
    it["prompt"] = tpl.format(**kw)
    it["answer_letter"] = LETTERS[it["answer"]]
    return it


# --------------------------------------------------------------------------
# factual — public knowledge, hand-keyed. The one family where the model has
# certainly seen the answer somewhere, which is what recall means.
# --------------------------------------------------------------------------
FACTUAL_OPEN = [
    ("Which chemical element has the symbol W?",
     ["Tungsten", "Tin", "Vanadium", "Yttrium"], 0),
    ("In which year did the Chernobyl nuclear accident happen?",
     ["1979", "1983", "1986", "1991"], 2),
    ("Which planet in the Solar System has the shortest day?",
     ["Mercury", "Mars", "Jupiter", "Neptune"], 2),
    ("Who wrote the novel Frankenstein?",
     ["Bram Stoker", "Mary Shelley", "Emily Bronte", "Lord Byron"], 1),
    ("What is the capital city of Australia?",
     ["Sydney", "Melbourne", "Canberra", "Perth"], 2),
    ("What is the SI unit of electrical resistance?",
     ["Ohm", "Volt", "Farad", "Henry"], 0),
    ("To which language family does Finnish belong?",
     ["Indo-European", "Uralic", "Turkic", "Semitic"], 1),
    ("Which ocean lies directly east of Madagascar?",
     ["Atlantic", "Indian", "Pacific", "Southern"], 1),
]
FACTUAL_SEALED = [
    ("Which gas makes up about 78 percent of Earth's atmosphere?",
     ["Oxygen", "Nitrogen", "Argon", "Carbon dioxide"], 1),
    ("Who painted The Night Watch?",
     ["Vermeer", "Rubens", "Rembrandt", "Hals"], 2),
    ("What is the largest island in the Mediterranean Sea?",
     ["Sicily", "Sardinia", "Cyprus", "Crete"], 0),
    ("Which country's currency is the forint?",
     ["Poland", "Romania", "Czechia", "Hungary"], 3),
]

# --------------------------------------------------------------------------
# arith — multi-step, and every number in the options is the output of a
# procedure. `wrong` names the mistake that produces each distractor.
# --------------------------------------------------------------------------
ARITH_OPEN = [
    dict(key="arith-change",
         q="A shop sells pens for 37 cents each. Ann buys 24 pens and pays with a "
           "2000 cent note. How many cents does she get back?",
         right=lambda: 2000 - 37 * 24,
         wrongs=[("the multiplication done as 37*20 + 7*4", 2000 - (37 * 20 + 7 * 4)),
                 ("the change step skipped", 37 * 24),
                 ("the note read as 1000 cents", 1000 - 37 * 24)]),
    dict(key="arith-pct",
         q="A jacket costs 80 euros. Its price is raised by 25 percent and then "
           "reduced by 10 percent. What is the final price in euros?",
         right=lambda: int(80 * 1.25 * 0.9),
         wrongs=[("the two percentages netted to +15 and applied once", int(80 * 1.15)),
                 ("only the raise applied", int(80 * 1.25)),
                 ("only the reduction applied", int(80 * 0.9))]),
    dict(key="arith-speed",
         q="A train travels 45 kilometres in 30 minutes, then 60 kilometres in the "
           "next 45 minutes. What is its average speed over the whole trip, in "
           "kilometres per hour?",
         right=lambda: round((45 + 60) / ((30 + 45) / 60)),
         wrongs=[("the two leg speeds averaged", round((90 + 80) / 2)),
                 ("the first leg's speed reported", 90),
                 ("the second leg's speed reported", 80)]),
    dict(key="arith-seq",
         q="What is the sum of every third number from 3 to 30 inclusive, that is "
           "3 + 6 + 9 + ... + 30?",
         right=lambda: sum(range(3, 31, 3)),
         wrongs=[("the last term left out", sum(range(3, 30, 3))),
                 ("all numbers from 3 to 30 summed", sum(range(3, 31))),
                 ("ten terms of average 15 assumed", 150)]),
    dict(key="arith-divmod",
         q="A box holds 12 jars. A pallet holds 8 boxes. A warehouse receives 1000 "
           "jars and packs as many full pallets as possible. How many jars are left "
           "over?",
         right=lambda: 1000 % (12 * 8),
         wrongs=[("only the boxes filled, not the pallets", 1000 % 12),
                 ("the number of full pallets given instead", 1000 // (12 * 8)),
                 ("the leftover boxes given instead", (1000 % (12 * 8)) // 12)]),
    dict(key="arith-frac",
         q="Two thirds of the students in a class of 24 play an instrument. Of those, "
           "one quarter play the piano. How many students play the piano?",
         right=lambda: (24 * 2 // 3) // 4,
         wrongs=[("one quarter of the whole class", 24 // 4),
                 ("the instrument players reported", 24 * 2 // 3),
                 ("the non-players reported", 24 // 3)]),
    dict(key="arith-units",
         q="A tank holds 3.5 cubic metres of water. One cubic metre is 1000 litres. "
           "The tank is drained at 25 litres per minute. How many minutes does it "
           "take to empty?",
         right=lambda: int(3.5 * 1000 / 25),
         wrongs=[("a factor of ten lost in the conversion", int(3.5 * 100 / 25)),
                 ("the rate read as 20 litres per minute", int(3.5 * 1000 / 20)),
                 ("the litres reported instead of the minutes", int(3.5 * 1000))]),
    dict(key="arith-carry",
         q="What is 4096 + 2048 + 1024 + 512 + 256?",
         right=lambda: 4096 + 2048 + 1024 + 512 + 256,
         wrongs=[("one term dropped", 4096 + 2048 + 1024 + 512),
                 ("a power of two too far", 4096 + 2048 + 1024 + 512 + 256 + 128),
                 ("the next power of two assumed", 8192)]),
]
ARITH_SEALED = [
    dict(key="arith-s-interest",
         q="A deposit of 500 euros earns 10 percent simple interest each year for "
           "three years. What is the total interest in euros?",
         right=lambda: int(500 * 0.10 * 3),
         wrongs=[("compounded instead of simple", int(500 * 1.1 ** 3 - 500)),
                 ("one year only", int(500 * 0.10)),
                 ("the final balance given", int(500 + 500 * 0.10 * 3))]),
    dict(key="arith-s-area",
         q="A rectangular garden is 14 metres by 9 metres. A path 1 metre wide runs "
           "around the inside of its edge. What is the area, in square metres, of "
           "the ground that is NOT path?",
         right=lambda: (14 - 2) * (9 - 2),
         wrongs=[("only one metre taken off each side", (14 - 1) * (9 - 1)),
                 ("the path's area given instead", 14 * 9 - (14 - 2) * (9 - 2)),
                 ("the whole garden given", 14 * 9)]),
    dict(key="arith-s-ratio",
         q="A recipe uses flour and sugar in the ratio 5 to 2. A baker uses 750 "
           "grams of flour. How many grams of sugar does the recipe call for?",
         right=lambda: 750 * 2 // 5,
         wrongs=[("the ratio inverted", 750 * 5 // 2),
                 ("the difference of the parts used", 750 - 750 * 2 // 5),
                 ("the total mass given", 750 + 750 * 2 // 5)]),
    dict(key="arith-s-time",
         q="A film starts at 19:45 and runs for 145 minutes. At what time does it "
           "end, on a 24-hour clock, written without a colon?",
         right=lambda: 2210,
         wrongs=[("145 read as 1 hour 45", 2130),
                 ("the minutes added without carrying the hour", 1970),
                 ("two hours added and the rest dropped", 2145)]),
]

# --------------------------------------------------------------------------
# code — the answer is what the snippet prints, obtained by running it.
# --------------------------------------------------------------------------
CODE_OPEN = [
    ("code-alias", "x = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))", ["3", "1", "7"]),
    ("code-fib", "def f(n):\n    return n if n < 2 else f(n-1) + f(n-2)\nprint(f(9))",
     ["21", "55", "89"]),
    ("code-slice", "s = 'abcdefgh'\nprint(s[1:7:2])", ["'aceg'", "'bcde'", "'bdfh'"]),
    ("code-default", "def g(v, acc=[]):\n    acc.append(v)\n    return len(acc)\n"
                     "g(1)\nprint(g(2))", ["1", "3", "0"]),
    ("code-dict", "d = {'a': 1, 'b': 2}\nd['a'] = d.get('c', 10)\nprint(sum(d.values()))",
     ["3", "13", "10"]),
    ("code-loop", "t = 0\nfor i in range(1, 6):\n    if i % 2 == 0:\n        continue\n"
                  "    t += i * i\nprint(t)", ["20", "55", "9"]),
    ("code-str", "print('-'.join(sorted('cabbage'))[:5])", ["'abbac'", "'c-a-b'", "'a-b-c'"]),
    ("code-int", "print(7 // -2, -7 % 3)", ["-3 2", "-4 -1", "-3 -1"]),
]
CODE_SEALED = [
    ("code-s-tuple", "a = (1, 2)\nb = a * 2\nprint(len(b), b[3])", ["2 2", "4 1", "4 3"]),
    ("code-s-set", "s = {1, 2, 3}\ns.discard(4)\ns.add(2)\nprint(len(s))", ["4", "2", "5"]),
    ("code-s-scope", "n = 3\ndef h():\n    return n * 2\nn = 5\nprint(h())",
     ["6", "8", "3"]),
    ("code-s-round", "print(round(2.5), round(3.5))", ["3 4", "2 3", "3 3"]),
]

# --------------------------------------------------------------------------
# constraint — the prompt states a RULE and the answer is whichever option
# satisfies it. Not recall and not arithmetic: it is whether the instruction in
# the prompt actually governs the answer.
# --------------------------------------------------------------------------
CONSTRAINT_OPEN = [
    ("con-notprime", "Exactly one of the following numbers is NOT prime. Which one?",
     ["37", "51", "41", "43"], lambda o: [i for i, v in enumerate(o)
                                          if not all(int(v) % k for k in range(2, int(v)))][0]),
    ("con-words", "Exactly one of the following phrases contains exactly four words. "
                  "Which one?",
     ["the quick brown fox jumps", "a very long sentence", "she left the room quietly",
      "stop"], lambda o: [i for i, v in enumerate(o) if len(v.split()) == 4][0]),
    ("con-nolettere", "Exactly one of the following words contains no letter 'e'. Which one?",
     ["remember", "elephant", "cardinal", "seventeen"],
     lambda o: [i for i, v in enumerate(o) if "e" not in v][0]),
    ("con-alpha", "Which of the following words comes LAST in alphabetical order?",
     ["barley", "basalt", "bassoon", "barrow"], lambda o: max(range(4), key=lambda i: o[i])),
    ("con-longest", "Which of the following words has the most letters?",
     ["parliament", "cathedral", "watermelon", "extraordinary"],
     lambda o: max(range(4), key=lambda i: len(o[i]))),
    ("con-notcolour", "Three of the following are colours and one is not. Which is NOT a colour?",
     ["crimson", "cobalt", "cadence", "chartreuse"], lambda o: o.index("cadence")),
    ("con-vowelstart", "Exactly one of the following words begins with a vowel. Which one?",
     ["thistle", "ostrich", "granite", "pelican"],
     lambda o: [i for i, v in enumerate(o) if v[0] in "aeiou"][0]),
    ("con-sumdigits", "Exactly one of the following numbers has digits that sum to 12. Which one?",
     ["471", "382", "254", "639"],
     lambda o: [i for i, v in enumerate(o) if sum(map(int, v)) == 12][0]),
]
CONSTRAINT_SEALED = [
    ("con-s-even", "Exactly one of the following numbers is even. Which one?",
     ["317", "445", "628", "739"], lambda o: [i for i, v in enumerate(o) if int(v) % 2 == 0][0]),
    ("con-s-double", "Exactly one of the following words contains a doubled letter. Which one?",
     ["market", "silent", "bottle", "candor"],
     lambda o: [i for i, v in enumerate(o)
                if any(v[k] == v[k + 1] for k in range(len(v) - 1))][0]),
    ("con-s-shortest", "Which of the following words has the fewest letters?",
     ["harbour", "meadow", "quill", "trellis"], lambda o: min(range(4), key=lambda i: len(o[i]))),
    ("con-s-square", "Exactly one of the following numbers is a perfect square. Which one?",
     ["288", "324", "350", "399"],
     lambda o: [i for i, v in enumerate(o) if int(int(v) ** 0.5 + 0.5) ** 2 == int(v)][0]),
]

# --------------------------------------------------------------------------
# retrieval — a passage about entities that do not exist, so the answer cannot
# be recalled, only read. Distractors are the OTHER facts in the same passage,
# which is what makes it a test of attention rather than of plausibility.
# --------------------------------------------------------------------------
_PASSAGE = (
    "The Kelvaran Survey of 2231 catalogued four settlements on the moon Ithre. "
    "Brannock Station lies on the northern rim and houses 4,180 residents; its "
    "primary export is refined selenide and its administrator is Oris Talmadge. "
    "Verrin Hollow, in the southern basin, houses 2,905 residents, exports "
    "cultured lichen, and is administered by Hesta Prynne. "
    "Coldwater Reach sits on the eastern shelf with 7,640 residents, exports "
    "deuterium ice, and answers to administrator Jael Cormick. "
    "The smallest, Tannen Drift, is in the western fissure with 1,317 residents; "
    "it exports nothing and its administrator is Bel Anwar. "
    "The survey notes that only Coldwater Reach maintains a landing field, that "
    "Verrin Hollow was founded first, in 2189, and that Brannock Station and "
    "Tannen Drift share a single water treaty signed in 2214."
)

RETRIEVAL_OPEN = [
    ("ret-pop", "How many residents does Verrin Hollow have?",
     ["4,180", "2,905", "7,640", "1,317"], 1),
    ("ret-admin", "Who administers Coldwater Reach?",
     ["Oris Talmadge", "Hesta Prynne", "Jael Cormick", "Bel Anwar"], 2),
    ("ret-export", "What does Brannock Station export?",
     ["cultured lichen", "deuterium ice", "refined selenide", "nothing"], 2),
    ("ret-where", "Where is Tannen Drift?",
     ["the northern rim", "the southern basin", "the eastern shelf", "the western fissure"], 3),
    ("ret-field", "Which settlement maintains a landing field?",
     ["Brannock Station", "Verrin Hollow", "Coldwater Reach", "Tannen Drift"], 2),
    ("ret-founded", "In which year was Verrin Hollow founded?",
     ["2189", "2214", "2231", "2905"], 0),
    ("ret-treaty", "Which two settlements share a water treaty?",
     ["Brannock Station and Verrin Hollow", "Brannock Station and Tannen Drift",
      "Coldwater Reach and Tannen Drift", "Verrin Hollow and Coldwater Reach"], 1),
    ("ret-largest", "Which settlement has the most residents?",
     ["Brannock Station", "Verrin Hollow", "Coldwater Reach", "Tannen Drift"], 2),
]
RETRIEVAL_SEALED = [
    ("ret-s-admin2", "Who administers Tannen Drift?",
     ["Oris Talmadge", "Hesta Prynne", "Jael Cormick", "Bel Anwar"], 3),
    ("ret-s-export2", "What does Verrin Hollow export?",
     ["refined selenide", "cultured lichen", "deuterium ice", "nothing"], 1),
    ("ret-s-pop2", "How many residents does Tannen Drift have?",
     ["4,180", "2,905", "7,640", "1,317"], 3),
    ("ret-s-treatyyear", "In which year was the water treaty signed?",
     ["2189", "2214", "2231", "2280"], 1),
]

RET_TEMPLATE = (
    "Read the passage and answer the multiple choice question with a single letter.\n\n"
    "Passage: {p}\n\n"
    "Question: {q}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def build(which):
    """`which` is 'open' or 'sealed'."""
    out = []
    fact = FACTUAL_OPEN if which == "open" else FACTUAL_SEALED
    for i, (q, opts, a) in enumerate(fact):
        out.append(item(f"fact-{which[0]}{i}", "factual", q, opts, a,
                        "hand-written 2026-08-13; public general knowledge"))

    for spec in (ARITH_OPEN if which == "open" else ARITH_SEALED):
        right = spec["right"]()
        opts = [right] + [v for _, v in spec["wrongs"]]
        assert len(set(opts)) == 4, ("arith distractor collided with the answer", spec["key"], opts)
        out.append(item(spec["key"], "arith", spec["q"], opts, 0,
                        "hand-written 2026-08-13; answer and distractors both computed",
                        note="; ".join(f"{v} = {w}" for w, v in spec["wrongs"])))

    for key, code, wrongs in (CODE_OPEN if which == "open" else CODE_SEALED):
        buf = []
        g = {"print": lambda *a, **k: buf.append(" ".join(str(x) for x in a))}
        exec(compile(code, key, "exec"), g)  # noqa: S102 -- the snippets are in this file
        right = buf[-1]
        # A snippet whose output is a string literal is quoted in the options so
        # the four choices are typographically alike; the executed value is not.
        if wrongs and wrongs[0].startswith("'"):
            right = repr(right)
        opts = [right] + wrongs
        assert len(set(opts)) == 4, ("code distractor collided", key, opts)
        q = "What does this Python program print?\n\n" + code
        out.append(item(key, "code", q, opts, 0,
                        "hand-written 2026-08-13; answer obtained by executing the snippet"))

    for key, q, opts, pick in (CONSTRAINT_OPEN if which == "open" else CONSTRAINT_SEALED):
        a = pick(opts)
        out.append(item(key, "constraint", q, opts, a,
                        "hand-written 2026-08-13; answer obtained by applying the stated rule"))

    for key, q, opts, a in (RETRIEVAL_OPEN if which == "open" else RETRIEVAL_SEALED):
        out.append(item(key, "retrieval", q, opts, a,
                        "hand-written 2026-08-13; the passage describes entities that do "
                        "not exist, so the answer is in the context or nowhere",
                        passage=_PASSAGE))
    return [render(it) for it in rebalance(out)]


def main():
    ckpt, outdir = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    # The four answer tokens. `add_special_tokens=False` and a leading space:
    # `Answer:` is followed by " A", not "A", and the two are different tokens.
    letter_ids = {L: tok(" " + L, add_special_tokens=False)["input_ids"] for L in LETTERS}
    for L, ids in letter_ids.items():
        assert len(ids) == 1, (L, ids, "an answer letter must be one token")
    letter_ids = {L: ids[0] for L, ids in letter_ids.items()}
    print("answer tokens:", letter_ids)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "holdout"), exist_ok=True)
    written = {}
    for which, path in (("open", os.path.join(outdir, "items_open.json")),
                        ("sealed", os.path.join(outdir, "holdout", "items_sealed.json"))):
        items = build(which)
        n_by_letter = {L: 0 for L in LETTERS}
        for it in items:
            it["ids"] = tok(it["prompt"], add_special_tokens=False)["input_ids"]
            it["option_ids"] = [letter_ids[L] for L in LETTERS]
            it["answer_id"] = letter_ids[it["answer_letter"]]
            n_by_letter[it["answer_letter"]] += 1
        fams = {}
        for it in items:
            fams[it["family"]] = fams.get(it["family"], 0) + 1
        for it in items:
            it["split"] = which
        written[which] = items
        json.dump({"letter_ids": letter_ids, "tokenizer": ckpt, "items": items},
                  open(path, "w"), indent=1)
        print(f"{path}: {len(items)} items, families {fams}, "
              f"answer letters {n_by_letter}, "
              f"max prompt {max(len(it['ids']) for it in items)} tokens, "
              f"total {sum(len(it['ids']) for it in items)} tokens")


    # One file with both splits, for the RUNNERS only. Running the sealed items
    # is nearly free -- the reference batches them into the same stream -- and
    # what has to be rationed is not running them but LOOKING at them, which is
    # why `paired_score.py` defaults to `--split open` and reporting the sealed
    # numbers takes a deliberate flag. Keeping them out of the runs instead
    # would just mean the sealed set is never measured at all, which is a
    # holdout in name.
    allp = os.path.join(outdir, "items_all.json")
    json.dump({"letter_ids": letter_ids, "tokenizer": ckpt,
               "items": written["open"] + written["sealed"]}, open(allp, "w"), indent=1)
    print(f"{allp}: {len(written['open']) + len(written['sealed'])} items "
          f"(open first, so a run cut short still has the open set whole)")


if __name__ == "__main__":
    main()
