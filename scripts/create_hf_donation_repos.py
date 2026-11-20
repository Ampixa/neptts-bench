"""Create the two private Hugging Face datasets that hold donated voice recordings.

    HF_TOKEN=<org-admin token> python scripts/create_hf_donation_repos.py --org Ampixa

Safe to re-run: existing repos are checked, never recreated or made public. Prints the
environment lines to paste into /etc/ampixa-arena/environment. Afterwards, create a
*fine-grained* token for the arena (write on these two repos only, read on
ampixa/rani-asr) and use that as HF_TOKEN on the VM, not the admin token used here.
"""

import argparse
import os
import sys

CARD = """---
license: other
license_name: ampixa-voice-donation-v1
pretty_name: {title}
private: true
task_categories: [text-to-speech, automatic-speech-recognition]
language: [ne]
---
# {title}

Recordings donated through https://ampixa.benchmarks.cloud/donate under consent
`voice-donation-v1`. **Private.** Access is limited to Ampixa staff named in the consent.
Donors may delete takes or withdraw at any time; deletions are erased from repo history.
Managed by `submission_app/donation_store.py`; do not edit files by hand.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--org", required=True)
    parser.add_argument("--benchmark-name", default="neptts-donations-benchmark")
    parser.add_argument("--open-name", default="neptts-donations-open")
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a token that can create repos in the organisation.")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repos = {}
    for key, name, title in (("BENCHMARK", args.benchmark_name, "NepTTS voice donations (benchmark anchor)"),
                             ("OPEN", args.open_name, "NepTTS voice donations (open prompts)")):
        repo = f"{args.org}/{name}"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(repo, repo_type="dataset").private:
            sys.exit(f"{repo} exists and is PUBLIC. Make it private before using it for donations.")
        files = api.list_repo_files(repo, repo_type="dataset")
        if "README.md" not in files:
            api.upload_file(path_or_fileobj=CARD.format(title=title).encode(), path_in_repo="README.md",
                            repo_id=repo, repo_type="dataset", commit_message="Dataset card")
        repos[key] = repo
        print(f"✓ {repo} (private)")
    print("\nAdd to /etc/ampixa-arena/environment:")
    print(f"NEPTTS_HF_DONATIONS_BENCHMARK_REPO={repos['BENCHMARK']}")
    print(f"NEPTTS_HF_DONATIONS_OPEN_REPO={repos['OPEN']}")
    print("HF_TOKEN=<fine-grained token: write on both repos, read on ampixa/rani-asr>")


if __name__ == "__main__":
    main()
