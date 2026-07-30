# Publish checklist (local)

Public GitHub lives in a **clean extract**, not the job-search mega-repo:

- Working copy for push: sibling `sonolab-public/` → https://github.com/althaafsn/sonolab
- Dev tree: this `sonolab/` folder (may still hold SPECFEM, datasets, `.venv`)

After code changes here, sync before pushing:

```bash
rsync -a --delete \
  --exclude '.venv/' --exclude 'datasets/' --exclude 'specfem2d-UT/' --exclude 'specfem3d/' \
  --exclude 'cursor-instructions-backup-*/' --exclude 'crates/*/target/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.git/' --exclude 'uv.lock' \
  ./ ../sonolab-public/
cd ../sonolab-public && git add -A && git status
```

**Public hygiene:** never name commercial ILI vendors or product lines in README, demos, commits, GitHub description, or LinkedIn copy.

## LinkedIn (Week 0)

1. Open [docs/demo/block8/LINKEDIN_DRAFT.md](docs/demo/block8/LINKEDIN_DRAFT.md)
2. Attach `orbit.mp4` (or `orbit.gif`) + optional stills
3. Post and Feature the repo for 2–3 weeks
4. Calendar: **next project post ~10–14 days later** (Week 2)

Do not publish multiple project dump posts in the same week.
