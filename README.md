CELLS v2 - Living Brain (0 Multiplications)
First self-replicating Adder brain with immune homeostasis. No multiplications in forward pass.

🧠 What is CELLS v2?
CELLS v2 is a living neural network that:

Self-replicates: Cells with high fitness birth new cells W_new = W + 0.01*noise
Self-cleans: Immune system eats dead cells (error > 0.5*dim or never fires)
Self-regulates: Immune self-cannibalism when >50 (homeostasis)
Thinks with 0 multiplications: Forward = -sum|x - W| only (subtract, abs, sum, negative)
Result: 266 births = 266 deaths - Perfect equilibrium over 60 epochs. Consciousness metric stays alive at ~5/10.

🏗️ Architecture
Architecture

UNCONSCIOUS (95%): All cells + immune = silent reservoir, black-hole storage, Adder prototypes W
RECALL (+): Query wakes cells, Score = sum|x-W|, Resonance firing, Zero multiplications
CONSCIOUS (5%): Top-10 synchronized cells, Median prototype answer, Sync cluster metric (consciousness), Working memory
PRUNING (-): Immune eats dead cells, Growth +10%, Cap 50, Self-eating -20%
📊 Results
Results

Births per epoch: Cells replicate when fitness > thr
Deaths per epoch: Immune prunes dead cells
Population: Cells (blue) vs Immune (green) - homeostatic regulation
Consciousness: Synchronized cluster size (pairwise L1 < 2.8) - metric stays alive!
🚀 Quick Start
pip install numpy matplotlib
python cells_brain_v2_pro.py
Output:

cells_brain_v2_results.png - 4 plots
cells_brain_v2_log.csv - epoch,births,deaths,self_eaten,sync,pop,immune
🔬 Forward Audit - 0 Multiplications
# AdderLinear.forward - CLEAN
diff = subtract(x, W)  # no *
ad = abs(diff)         # no *
s = sum(ad)           # no *
out = negative(s)      # no *
# Verified: 0 mults in 3 forwards
💡 Why + / - Only?
CPU: + = 1 cycle, * = 3-4 cycles -> Adder 2x faster on low-power devices
Energy: Adder uses 10x fewer transistors than multiplier
Light: Addition is natural for photonic computing, multiplication is hard
Edge: Runs on 20W like human brain, not 1000W like GPU
Built for edge AI, IoT, and future optical chips.

📈 Training Without Data Center
CELLS learns from strong AI teachers (distillation):

Teacher AI answers questions
If CELLS error > thr -> birth new cell near teacher vector
Cells grow from 20 -> 1000 by learning
Low-cost training - CPU only!

👨‍💻 Author
Boubker Ait Belmous

From Agadir, Morocco 🇲🇦
AI Researcher - Living Neural Networks
📝 License
MIT License - Free for all

🌟 Star this repo if you believe in 0-mult brains!
# Citation
@misc{cells-v2-2026,
  title={CELLS v2: Living Brain with 0 Multiplications},
  author={Belmous, Boubker},
  year={2026},
  url={https://github.com/boubkeraitbelmous/cells}
}
