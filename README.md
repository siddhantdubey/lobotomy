> **Disclaimer**: This repo is AI generated. Claude wrote the code and all of the writing, including this README, in my voice. The sketch checker was tested against real browser results, but the full script hasn't been run end to end yet. Read it before you trust it.

# Lobotomizing An AI

In the video I promised a guide on lobotomizing an AI, so here it is. This is the smallest version I could make of one piece of my [Taste Transplant](https://lab.siddubey.com/projects/taste-transplant/) experiment: find the neurons that are busy when Qwen writes code that crashes, and turn them off.

The whole thing is two files. `lobotomy.py` does the model stuff and `check.js` decides whether a sketch crashed. You can read both in about ten minutes.

## What Was Breaking

I asked Qwen3-8B to paint "{subject} in watercolour" with p5.js and the [p5.brush](https://github.com/acamposuribe/p5.brush) library. Only 118 of the first 384 sketches ran at all. Half of all sketches were dead from their very first line, and almost always the same way:

```js
let brush;

function setup() {
  brush = createBrush();
  ...
```

p5.brush already gives you a global called `brush`. Writing `let brush` hides it, and `createBrush()` doesn't exist, so the sketch dies before it paints anything. The model picked up a bad habit from somewhere and repeats it constantly. That's a good target for surgery: a specific, repeatable mistake that you can catch automatically.

## How It Works

1. Ask Qwen for 128 sketches (64 subjects, 2 each).
2. Run them through `check.js` and split them into "crashed" and "worked".
3. Run every sketch back through the model and record how hard each MLP neuron fires while it writes. Find the 32 neurons that fire much more in the crashed pile than in the working pile.
4. Turn them off. A neuron's output is one column of its layer's `down_proj` weight, so we set that column to zero. The neuron still fires, but nothing downstream can hear it. That's the lobotomy, and it's one line:

   ```python
   model.model.layers[layer].mlp.down_proj.weight[:, neuron] = 0
   ```

5. Ask for 64 new sketches on 32 subjects the model never saw in step 1, and count the crashes. Do it three times with the same random seed: untouched, lobotomized, and with 32 *random* neurons from the same layers turned off.

Step 5's random row is the one people skip, and it's the one that matters most. More on that below.

## Running It

You need Python, Node (any recent version) and a GPU. The default model is Qwen3-1.7B, which should fit on a 16 GB card like Colab's free T4. Swap `MODEL` at the top of `lobotomy.py` for something else if you want, as long as it's a Qwen or Llama style model with `model.model.layers[i].mlp.down_proj`.

```bash
pip install -r requirements.txt
python lobotomy.py
```

It prints three lines at the end, one per row, with how many sketches crashed and the most common errors. Everything, including every sketch and the neurons it picked, goes into `results/`.

## Reading The Results

Compare the lobotomized row against the random row, not just against the untouched row.

When I did this for real, I zeroed the 64 quietest neurons in every layer of Qwen3-8B (about 2% of them) and the share of sketches that ran went from 31% to 43%. Sounds like I found the bad neurons, right? Nope. Zeroing a different 2%, like the medium-busy neurons, did about the same thing. Only the middle third of the layers mattered, and inside it, which neurons you picked hardly did. The `createBrush` habit sits on a fork in the first line of code, and nudging the model almost anywhere tips it the other way. If your lobotomized row and your random row look the same, you didn't find anything special. You just shook the model.

Also, 64 sketches per row isn't a lot. A difference of three or four crashes is noise. Bump `SAMPLES_PER_SUBJECT` if you want to be sure.

## What Actually Worked

The fix that held up in the real experiment wasn't a cut. It was the opposite. Claude Opus 5.5 hand-wrote 4 new neurons into layer 33 of Qwen3-8B (no training, just directions computed from activations) that act like vetoes, things like "don't declare a top-level variable here." On subjects it had never seen, the share of sketches that rendered something went from 34% to 82%, and the `createBrush` mistake went to zero. Random neurons of the same size in the same spots broke the model completely: 0 of 128 sketches ran.

Then we lobotomized *that*. Turning off just the one "no top-level declaration" neuron brought the `createBrush` mistake back from 0% to 60% of sketches, and the rest of the edit kept doing its job. That's the cleanest evidence I have that a specific neuron was doing a specific job.

So turning neurons off is a great way to find out what a part of the network does. It's a less reliable way to fix things. This repo shows the turning-off half because it's the easiest place to start.

## The Fine Print

- `check.js` is a fake browser, not a real one. It gives the sketch a pretend p5.js and p5.brush where everything exists but draws nothing, then sees if it throws. On 4,352 Qwen3-8B sketches from my experiment it agreed with real headless Chrome on 4,324. The 28 misses were bad colour strings that only real p5 notices.
- This only fixes crashing. It doesn't make the paintings good. A sketch that runs can still be an ugly brown blob, and teaching Qwen taste was the hard part of the experiment.
- Step 3 averages each neuron over the whole reply. That's the crudest choice. The real decision happens in the first few tokens of the code, so averaging only over those would probably find sharper neurons. Try it.
- The full write-ups, with the controls and everything that didn't work, are on [my lab notebook](https://lab.siddubey.com/projects/taste-transplant/).
