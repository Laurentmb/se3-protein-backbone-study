# Publish this project on GitHub

## GitHub Desktop

1. Open GitHub Desktop and sign in to your GitHub account.
2. Choose **File → Add local repository** and select this project's root folder
   (the folder containing `README.md`, `scripts/` and `results/`).
3. Review the selected files. Create a first commit with the summary
   `Initial release: reproducible SE(3) sampling study`.
4. Click **Publish repository**. Suggested name: `se3-protein-backbone-study`.
5. For a public portfolio, clear **Keep this code private**, then publish to
   your personal account.
6. Open the repository on GitHub and check the README, figures and Actions tab.
   The configured CPU checks run after publication; no GPU or ESM Atlas access
   is needed by those checks.

The prepared local folder is already a Git repository with the project files
staged. If using the ZIP on another machine, extract it first. Its `.git` folder
is omitted: create a local repository there before publishing. Publish the
extracted contents as source files so GitHub can display the README and code.

Suggested repository description:

> Reproducible SE(3) protein backbone sampling study: Euler vs Lie midpoint,
> ProteinMPNN sequence design and ESMFold self-consistency evaluation.

Suggested topics: `protein-design`, `se3`, `flow-matching`, `pytorch`,
`numerical-methods`, `reproducible-research`.

## Command-line alternative

Create an empty repository on GitHub, then run these commands from this folder.
Replace `YOUR-USERNAME` with your account name:

```bash
git add .
git commit -m "Initial release: reproducible SE(3) sampling study"
git remote add origin https://github.com/YOUR-USERNAME/se3-protein-backbone-study.git
git push -u origin main
```

GitHub authentication is handled by GitHub Desktop or Git's credential manager.
Downloaded models and local runs are excluded by `.gitignore`.

Official instructions:
[add a local repository](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-a-repository-from-your-local-computer-to-github-desktop)
and [publish an existing project](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop).
