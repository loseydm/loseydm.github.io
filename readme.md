# Personal website

Static personal site for Darby Losey. Plain HTML + CSS — no build step, no dependencies.

## Structure

```
index.html                  landing page (edit the bio text here)
assets/css/site.css         site-wide styling; colors live in the :root block
assets/img/                 headshot and other images
paper-lab/index.html        "Darby's Paper Lab" landing page (entry cards)
paper-lab/hippo/index.html  interactive HiPPO essay (article text lives here)
paper-lab/hippo/hippo.js    HiPPO demo: math (ODE integration) + rendering
paper-lab/hippo/hippo.css   article + demo styling
paper-lab/mamba/index.html  interactive Mamba (selective SSM) essay, four demos
paper-lab/mamba/mamba.js    demo 1: selective-scan recurrence + rendering
paper-lab/mamba/memory.js   demo 2: vector B,C as an associative memory (induction)
paper-lab/mamba/scan.js     demo 3: parallel (Hillis-Steele) scan ladder
paper-lab/mamba/cost.js     demo 4: generation cost, KV cache vs fixed state
paper-lab/mamba/mamba.css   article + all demo styling
.nojekyll                   tells GitHub Pages to serve files as-is
```

`mamba_123_tutorial.pdf` (the "Mamba, from Zero to Hero" write-up) is intentionally
kept out of the site for now — it is git-ignored, so it stays in this folder locally
but is not published. When it is ready, remove it from `.gitignore` and re-add the
links in `paper-lab/index.html` and `paper-lab/hippo/index.html`.

The HiPPO page pulls KaTeX (math rendering) and the Inter/Lora fonts from
CDNs; everything else is self-contained and runs entirely in the browser.

## Local preview

```
python -m http.server 8000
```

Then open <http://localhost:8000>.

## Deploying to GitHub Pages

Push the contents of this folder to the root of a repository named
`<username>.github.io`. GitHub Pages serves it automatically at
`https://<username>.github.io`.

```
git init
git add .
git commit -m "New personal site"
git branch -M main
git remote add origin https://github.com/<username>/<username>.github.io.git
git push -u origin main
```

If the repo already exists with the old site, push to a branch and merge
rather than force-pushing over the existing history.
