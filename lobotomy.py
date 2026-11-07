"""
Lobotomize a small Qwen so it stops writing p5.brush sketches that crash.

  1. Ask Qwen for a bunch of watercolour sketches.
  2. Run each one through check.js and sort them into "crashed" and "worked".
  3. Look inside the model while it writes, and find the neurons that are busier
     in the crashed sketches than in the ones that worked.
  4. Turn those neurons off by zeroing their output weights.
  5. Ask for new sketches, on subjects it never saw in step 1, and count crashes again.
     Do the same with randomly chosen neurons turned off, so we know whether
     *which* neurons we cut actually matters.

python lobotomy.py
"""

import json
import random
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
NEURONS_TO_CUT = 32
SAMPLES_PER_SUBJECT = 2
MAX_NEW_TOKENS = 1500
BATCH_SIZE = 16
OUT = Path("results")

SYSTEM = """You write p5.js sketches that paint with the p5.brush library (version 1.1.4) on p5.js 1.11.

Rules:
- Reply with one complete sketch in a single ```javascript code block and nothing else.
- The sketch makes one static 512x512 image. In setup(), call createCanvas(512, 512, WEBGL), then translate(-width / 2, -height / 2) so that (0, 0) is the top-left corner. Paint everything in setup() and call noLoop().
- Make every mark with these brush methods: brush.fill(color, opacity), brush.bleed(strength, direction), brush.fillTexture(textureStrength, borderIntensity), brush.noStroke(), brush.polygon(pointsArray), brush.circle(x, y, radius, r), brush.set(brushName, color, weight), brush.spline(array_points, curvature).
- Plain p5.js is available for background(), translate(), random(), noise(), color(), lerpColor(), map() and maths."""

# Subjects we use to find the neurons.
FIND_SUBJECTS = [
    "a pink peony", "a white lily", "a yellow tulip", "a sprig of eucalyptus", "a sleeping cat", "an elephant",
    "a red fox", "a tiger", "a pair of swans", "a flamingo", "a peacock", "a kingfisher", "waves breaking on rocks",
    "a whale's tail", "a lighthouse at dusk", "a beach at sunset", "a misty valley at dawn", "a lone oak tree",
    "a desert with cacti", "an autumn maple tree", "a castle on a hill", "a windmill", "a row of terraced houses",
    "a stone bridge", "a pot of tea", "a basket of strawberries", "a lemon on a plate", "a stack of pancakes",
    "a hot air balloon", "an old typewriter", "a stack of books", "a violin", "a blue iris", "a peach hibiscus",
    "a red poppy", "a dandelion clock", "a giraffe", "a frog on a lily pad", "a hare", "a sheepdog", "an owl",
    "a hummingbird", "a crow", "a robin on a branch", "a waterfall", "a fishing village harbour",
    "a river through a forest", "a jellyfish", "a snowy pine forest", "a mountain range", "a canyon",
    "a field of wheat", "a Paris street cafe", "a red barn", "a Japanese pagoda", "a cottage with a garden",
    "a cup of coffee", "a slice of watermelon", "a sushi platter", "a pear", "an umbrella in the rain",
    "a pair of boots", "a vase of wildflowers", "a seashell",
]

# Subjects we only use at the end, to test the cut. The model never sees these in step 1.
TEST_SUBJECTS = [
    "a bunch of lavender", "a cherry blossom branch", "a sunflower", "a lotus flower", "a hedgehog", "a brown bear",
    "a horse grazing", "a deer in a clearing", "a seagull", "a puffin", "a heron", "a sparrow",
    "a sailboat on a calm sea", "a coral reef", "a koi pond", "a rowing boat on a lake", "rolling green hills",
    "a lavender field in Provence", "a birch grove", "a volcano", "a cathedral", "a lighthouse keeper's cottage",
    "a Venice canal", "a greenhouse", "a croissant", "a bowl of cherries", "a jar of honey", "a loaf of bread",
    "a bicycle", "a candle", "a teapot and cups", "a paper lantern",
]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    return tokenizer, model


def prompt_ids(tokenizer, subject):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{subject} in watercolour"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text)["input_ids"]


@torch.no_grad()
def write_sketches(tokenizer, model, subjects, seed):
    """Ask for SAMPLES_PER_SUBJECT sketches per subject. Same seed in, same sketches out."""
    jobs = [(s, i) for s in subjects for i in range(SAMPLES_PER_SUBJECT)]
    sketches = []
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start:start + BATCH_SIZE]
        prompts = [prompt_ids(tokenizer, s) for s, _ in batch]
        padded = tokenizer.pad({"input_ids": prompts}, return_tensors="pt").to(model.device)
        torch.manual_seed(seed + start)
        out = model.generate(**padded, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.8,
                             pad_token_id=tokenizer.pad_token_id)
        for (subject, i), prompt, row in zip(batch, prompts, out):
            reply = row[padded["input_ids"].shape[1]:].tolist()
            while reply and reply[-1] == tokenizer.pad_token_id:
                reply.pop()
            text = tokenizer.decode(reply, skip_special_tokens=True)
            sketches.append({"id": f"{subject} #{i}", "subject": subject, "prompt": prompt, "reply": reply,
                             "text": text, "code": get_code(text)})
        print(f"  wrote {len(sketches)}/{len(jobs)}")
    return sketches


def get_code(text):
    match = re.search(r"```(?:javascript|js)?\s*\n(.*?)(```|$)", text, re.S)
    return match.group(1) if match else text


def check(sketches):
    """Run every sketch through check.js. Adds "ok" and "error" to each one."""
    lines = "".join(json.dumps({"id": s["id"], "code": s["code"]}) + "\n" for s in sketches)
    out = subprocess.run(["node", "check.js"], input=lines, capture_output=True, text=True, check=True).stdout
    verdicts = {v["id"]: v for v in map(json.loads, out.splitlines())}
    for s in sketches:
        s["ok"], s["error"] = verdicts[s["id"]]["ok"], verdicts[s["id"]]["error"]
    return sketches


@torch.no_grad()
def neuron_activity(model, sketches):
    """For each sketch, how busy each MLP neuron was, on average, while writing the reply.

    The "neurons" are the inputs to each layer's down_proj, the point where the MLP
    has done its thinking and is about to write back into the model.
    Returns a tensor shaped [sketches, layers, neurons].
    """
    layers = model.model.layers
    current = {}

    def grab(layer_index):
        def hook(module, inputs):
            current[layer_index] = inputs[0][0]
        return hook

    hooks = [layer.mlp.down_proj.register_forward_pre_hook(grab(i)) for i, layer in enumerate(layers)]
    activity = []
    for s in sketches:
        ids = torch.tensor([s["prompt"] + s["reply"]], device=model.device)
        model(ids)
        start = len(s["prompt"])
        activity.append(torch.stack([current[i][start:].float().mean(0).cpu() for i in range(len(layers))]))
    for h in hooks:
        h.remove()
    return torch.stack(activity)


def pick_neurons(activity, crashed, k):
    """The k neurons that are much busier in crashed sketches than in working ones."""
    bad, good = activity[crashed], activity[~crashed]
    spread = activity.std(0) + 1e-6
    score = (bad.mean(0) - good.mean(0)) / spread
    n_neurons = activity.shape[2]
    top = score.flatten().topk(k).indices.tolist()
    return [(i // n_neurons, i % n_neurons, round(score.flatten()[i].item(), 3)) for i in top]


def random_neurons_like(picked, n_neurons, seed=0):
    """Same number of neurons in the same layers, but chosen at random."""
    rng = random.Random(seed)
    return [(layer, rng.randrange(n_neurons), None) for layer, _, _ in picked]


@torch.no_grad()
def cut(model, neurons):
    """Zero the output weights of these neurons. They still fire, but nothing downstream hears them.
    Returns a function that puts them back."""
    saved = []
    for layer, neuron, _ in neurons:
        weight = model.model.layers[layer].mlp.down_proj.weight
        saved.append((weight, neuron, weight[:, neuron].clone()))
        weight[:, neuron] = 0

    @torch.no_grad()
    def undo():
        for weight, neuron, column in reversed(saved):
            weight[:, neuron] = column
    return undo


def summary(name, sketches):
    crashed = [s for s in sketches if not s["ok"]]
    errors = {}
    for s in crashed:
        short = s["error"].split("\n")[0][:60]
        errors[short] = errors.get(short, 0) + 1
    top = sorted(errors.items(), key=lambda x: -x[1])[:3]
    print(f"{name:<22} crashed {len(crashed):>3}/{len(sketches)}   top errors: {top}")
    return {"crashed": len(crashed), "n": len(sketches), "errors": errors}


def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1))


def main():
    tokenizer, model = load_model()

    print("1. Writing sketches to learn from")
    sketches = check(write_sketches(tokenizer, model, FIND_SUBJECTS, seed=1))
    crashed = torch.tensor([not s["ok"] for s in sketches])
    summary("before", sketches)
    save("find_sketches.json", [{k: s[k] for k in ("id", "code", "ok", "error")} for s in sketches])
    if crashed.all() or not crashed.any():
        raise SystemExit("Need some sketches that crash and some that work. Try a different model or more samples.")

    print("2. Watching the neurons")
    activity = neuron_activity(model, sketches)
    picked = pick_neurons(activity, crashed, NEURONS_TO_CUT)
    control = random_neurons_like(picked, activity.shape[2])
    save("neurons.json", {"picked": picked, "random": control})
    print("  picked (layer, neuron, score):", picked[:8], "...")

    print("3. Testing on new subjects")
    results = {}
    for name, neurons in [("untouched", []), ("lobotomized", picked), ("random neurons cut", control)]:
        undo = cut(model, neurons)
        test = check(write_sketches(tokenizer, model, TEST_SUBJECTS, seed=2))
        undo()
        results[name] = summary(name, test)
        save(f"test_{name.replace(' ', '_')}.json", [{k: s[k] for k in ("id", "code", "ok", "error")} for s in test])
    save("summary.json", results)


if __name__ == "__main__":
    main()
