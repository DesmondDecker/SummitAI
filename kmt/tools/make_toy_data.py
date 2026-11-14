"""Create SYNTHETIC toy languages to verify the whole pipeline works on your machine.
   python tools/make_toy_data.py --out toy_data   then   python train.py --data toy_data --out toy_ckpt --steps 3000 --d 96 --layers 3 --ctx 48 --merges 150 --lr 3e-3
Expected: ~100% held-out translation chrF. (These are NOT real languages.)"""
import argparse, os, random

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="toy_data"); a = ap.parse_args()
random.seed(0)
subj, verb = "dog cat man child woman bird".split(), "sees eats likes finds".split()
obj, adj = "house bread water fish tree".split(), "big small old".split()
words = ["the"] + subj + verb + obj + adj


def lexicon(seed):
    r, syl, used, m = random.Random(seed), "ba ko mi tu le sa no ge fu da ri po".split(), set(), {}
    for w in words:
        while True:
            x = "".join(r.choice(syl) for _ in range(2))
            if x not in used:
                used.add(x); m[w] = x; break
    return m


combos = [("the", x, s, v, "the", o) for x in adj for s in subj for v in verb for o in obj]
random.shuffle(combos); pairs = combos[:260]
tr = lambda ws, m: " ".join(m[w] for w in ws)
os.makedirs(os.path.join(a.out, "parallel"), exist_ok=True)
for name, m in (("mende", lexicon(1)), ("temne", lexicon(2))):
    open(f"{a.out}/parallel/{name}-en.tsv", "w").write("".join(tr(s, m) + "\t" + " ".join(s) + "\n" for s in pairs))
    open(f"{a.out}/{name}.txt", "w").write("\n".join(tr(random.choice(combos), m) for _ in range(150)) + "\n")
open(f"{a.out}/en.txt", "w").write("\n".join(" ".join(random.choice(combos)) for _ in range(150)) + "\n")
rows = [random.choice(combos) for _ in range(30)]
open(f"{a.out}/intents.tsv", "w").write("".join(f"en\t{' '.join(s)}\t{'about_dog' if 'dog' in s else 'other'}\n" for s in rows))
# reading-comprehension task: answer a question using facts given in the context
nouns, quals = subj, adj
qa = []
for _ in range(500):
    n1, n2 = random.sample(nouns, 2); a1, a2 = random.sample(quals, 2)
    ctx = f"the {n1} is {a1} . the {n2} is {a2} ."
    n, ans = random.choice([(n1, a1), (n2, a2)])
    qa.append(f"en\t{ctx}\twhat is the {n} ?\t{ans}\n")
open(f"{a.out}/qa.tsv", "w").write("".join(qa))
print("toy data in", a.out)
