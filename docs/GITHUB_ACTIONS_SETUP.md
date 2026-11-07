# GitHub-to-Modal automation - prepared, not executed

Workflow: [.github/workflows/modal-deploy.yml](../.github/workflows/modal-deploy.yml).
It currently supports only manual workflow_dispatch on main, using Python 3.11,
Ubuntu 24.04 and pinned official Action commit IDs. Modal SDK is pinned to 1.6.3.
It uses the existing app trident-depth-t69 and existing private Volume trident-t69-private-models.
It never calls the checkpoint uploader, creates a Volume, or uploads datasets/weights.

## Owner setup after publication review

1. Review LOCAL_HISTORY_VERIFICATION.md for the completed local cleanup and approved
   publication scope. Original evidence remains private; fresh-clone after the coordinated
   rewritten-history update. Deployment approval is separate from publication approval.
2. In GitHub repository Settings > Secrets and variables > Actions, add repository secrets
   **MODAL_TOKEN_ID** and **MODAL_TOKEN_SECRET** for the intended Modal workspace. These
   are the authentication environment variable names used by the installed Modal SDK.
   Obtain credentials privately through your Modal account; do not put values in chat,
   commands, reports, workflow source or Actions logs. No extra GitHub PAT is required.
3. In Settings > Environments, configure **production**, restrict deployment to main and
   enable owner review/protection where the account plan supports it. Merely naming an
   environment in YAML does not configure required reviewers. Confirm the protection in UI.
4. Leave repository Actions variable **MODAL_DEPLOY_ENABLED** unset until you explicitly
   approve the first production deployment. Unset/false makes the deployment job skip,
   and no push trigger is configured. Add the variable with exact value **true** only after
   secrets and owner approval are in place, then manually dispatch the workflow on main.
5. Approve the environment job only after explicit deployment authorization. Automatic
   deployment on main pushes is deferred to a separately approved integration phase.
   Use branch protection/review rules for main. Set the variable false to block manual
   deployment jobs without changing the current live application.

GitHub environment review protects deployment jobs, but must be configured by the owner:
[official environment documentation](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).

## What the job does

Install pinned deployment and app-smoke requirements; fail if either authentication secret
is missing; run scripts.modal_preflight to read the existing Function URL and stream-hash
the existing checkpoint in memory; deploy scripts/modal_deploy.py; run the existing synthetic
HTTP checker with --require-learned; upload only JSON verification and identity artifacts.
The preflight creates nothing and writes no checkpoint bytes to disk. A missing app, Volume,
checkpoint or mismatched URL/hash fails before deployment. The deployment module now uses
create_if_missing=False, so it cannot silently create a replacement private Volume.
Classical fallback remains available in the application itself, but CI requires learned health.

The approved checkpoint remains SHA256
09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd.
Only the hash is public: checkpoint bytes remain private. The resource budget is unchanged
(2 CPU cores / 4 GiB / no GPU / one container / 30 s idle scale-down). Hosting usage is billable.
There is no automatic rollback claim; a failed post-deploy check is a failed workflow requiring
investigation. GitHub Actions runner usage is subject to the owner's account allowances.

## Verification gates

Local validation: YAML parses; triggers, opt-in, environment, read-only permissions and pinned
Action IDs checked; read-only live preflight passed. This is not a successful Actions run.
After the first approved real run, record its run URL, source commit SHA, Modal application
identity, checkpoint hash and uploaded smoke artifacts. Recheck browser loading/errors and
downloads on the existing public URL. Until that run succeeds, **automatic deployment is unverified**.
No workflow, secret, environment or repository setting was changed remotely during preparation.
