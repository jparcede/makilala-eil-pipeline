# Pushing this repository

The initial commit is already made on branch `main` (34 files, ~366 KB).
What remains needs your GitHub credentials, so it has to be run by you.

Open Terminal on the Mac and `cd` to this folder:

    cd ~/Downloads/makilala_data/makilala-eil-pipeline

## Option A — with the GitHub CLI (simplest)

    brew install gh          # if not already installed
    gh auth login            # browser flow, once
    gh repo create makilala-eil-pipeline --private --source=. --push

That creates the repository under your account and pushes in one step.
To put it under a Caraga State organisation instead:

    gh repo create <org>/makilala-eil-pipeline --private --source=. --push

## Option B — without gh

1. Create an empty **private** repository on github.com named
   `makilala-eil-pipeline`. Do not add a README, .gitignore or licence --
   this repository already has them and GitHub's would conflict.

2. Then:

       git remote add origin https://github.com/<you>/makilala-eil-pipeline.git
       git push -u origin main

   GitHub will ask for a personal access token rather than a password
   (Settings -> Developer settings -> Personal access tokens, `repo` scope).

## When the paper is accepted

1. Make the repository public (Settings -> General -> Danger Zone).
2. Enable the Zenodo integration at https://zenodo.org/account/settings/github/
   and switch this repository on.
3. Cut a release on GitHub, tag `v1.0.0` to match `CITATION.cff`.
   Zenodo mints the DOI automatically from `.zenodo.json`.
4. Paste that DOI over the `XX.XXXX/zenodo.XXXXXXX` placeholder in the
   manuscript's Data availability statement, and the repository URL beside it.

## Note on the commit author

Committed as "Jayrold P. Arcede <jparcede@carsu.edu.ph>", set locally for this
repository only (from CITATION.cff). To change it:

    git config user.name  "..."
    git config user.email "..."
    git commit --amend --reset-author --no-edit

The commit message carries a Co-Authored-By trailer for the assistant that wrote
the v10 harness and the tests. Remove it with the same `--amend` if you would
rather it were not in the history.
