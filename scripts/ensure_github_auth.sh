#!/usr/bin/env bash
set -Eeuo pipefail
# Do not inherit xtrace from an invoking agent or shell. Authentication commands
# must never be echoed with environment-derived credential material.
set +x

readonly GITHUB_HOST="github.com"
readonly GIT_REMOTE="origin"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/ensure_github_auth.sh" >&2
    exit 2
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

github_cli_config_dir() {
    if [[ -n "${GH_CONFIG_DIR:-}" ]]; then
        printf '%s\n' "$GH_CONFIG_DIR"
    elif [[ -n "${XDG_CONFIG_HOME:-}" ]]; then
        printf '%s/gh\n' "$XDG_CONFIG_HOME"
    else
        printf '%s/.config/gh\n' "$HOME"
    fi
}

github_com_token_is_plaintext() {
    local hosts_file="$1"
    [[ -f "$hosts_file" ]] || return 1

    # Inspect only YAML key names. Never print or capture the oauth_token value.
    awk '
        /^[^[:space:]][^:]*:[[:space:]]*$/ {
            in_github = ($0 == "github.com:")
        }
        in_github && /^[[:space:]]+oauth_token:[[:space:]]*/ {
            found = 1
        }
        END { exit(found ? 0 : 1) }
    ' "$hosts_file"
}

resolve_github_repository() {
    local remote_url="$1"
    local path owner repository extra

    case "$remote_url" in
        https://github.com/*)
            path="${remote_url#https://github.com/}"
            ;;
        http://github.com/*)
            path="${remote_url#http://github.com/}"
            ;;
        git@github.com:*)
            path="${remote_url#git@github.com:}"
            ;;
        ssh://git@github.com/*)
            path="${remote_url#ssh://git@github.com/}"
            ;;
        git://github.com/*)
            path="${remote_url#git://github.com/}"
            ;;
        *)
            return 1
            ;;
    esac

    path="${path%%\?*}"
    path="${path%%\#*}"
    path="${path%/}"
    path="${path%.git}"
    IFS=/ read -r owner repository extra <<<"$path"
    [[ -n "$owner" && -n "$repository" && -z "${extra:-}" ]] || return 1
    [[ "$owner" =~ ^[A-Za-z0-9_.-]+$ ]] || return 1
    [[ "$repository" =~ ^[A-Za-z0-9_.-]+$ ]] || return 1
    printf '%s/%s\n' "$owner" "$repository"
}

if ! command -v gh >/dev/null 2>&1; then
    echo "[github-auth] GitHub CLI (gh) is missing from the Brev vscode container." >&2
    echo "[github-auth] From the Brev host, install it with:" >&2
    echo "  docker exec --user root vscode bash -lc 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y gh'" >&2
    echo "[github-auth] Then return to this clone and rerun:" >&2
    echo "  bash scripts/ensure_github_auth.sh" >&2
    exit 3
fi

if gh api --hostname "$GITHUB_HOST" user --jq '.login' >/dev/null 2>&1; then
    echo "[github-auth] Reusing an existing GitHub CLI login for $GITHUB_HOST."
else
    if [[ -n "${GH_TOKEN:-}" || -n "${GITHUB_TOKEN:-}" ]]; then
        echo "[github-auth] GH_TOKEN or GITHUB_TOKEN is set, but GitHub CLI cannot authenticate with it." >&2
        echo "[github-auth] Unset the invalid environment credential before starting browser/device login:" >&2
        echo "  unset GH_TOKEN GITHUB_TOKEN" >&2
        echo "  bash scripts/ensure_github_auth.sh" >&2
        exit 5
    fi

    echo "[github-auth] No valid GitHub CLI login is available for $GITHUB_HOST."
    echo "[github-auth] Human authorization is required through GitHub's browser/device flow."

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "[github-auth] Open an interactive Brev terminal, tell the human authorization is required, and run:" >&2
        echo "  bash scripts/ensure_github_auth.sh" >&2
        exit 4
    fi

    echo "[github-auth] The next prompt may display a one-time URL and device code in this terminal."
    echo "[github-auth] Ask the human to approve that code directly at GitHub; it is ephemeral, not a reusable token."
    gh auth login --hostname "$GITHUB_HOST" --git-protocol https --web
    gh api --hostname "$GITHUB_HOST" user --jq '.login' >/dev/null 2>&1 || fail \
        "GitHub CLI authentication did not complete successfully."
fi

active_account="$(gh api --hostname "$GITHUB_HOST" user --jq '.login' 2>/dev/null)" || fail \
    "GitHub accepted a credential, but the active account could not be identified."
[[ "$active_account" =~ ^[A-Za-z0-9-]+$ ]] || fail \
    "GitHub returned an invalid active-account name."
echo "[github-auth] Active GitHub account: $active_account"

auth_source="stored"
if [[ -n "${GH_TOKEN:-}" ]]; then
    auth_source="environment"
    echo "[github-auth] Credential source: GH_TOKEN (transient environment credential; not stored by this script)."
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    auth_source="environment"
    echo "[github-auth] Credential source: GITHUB_TOKEN (transient environment credential; not stored by this script)."
else
    config_dir="$(github_cli_config_dir)"
    hosts_file="$config_dir/hosts.yml"
    if github_com_token_is_plaintext "$hosts_file"; then
        auth_source="plaintext"
        echo "[github-auth] WARNING: GitHub CLI is storing this login in plaintext at $hosts_file."
        echo "[github-auth] Anyone who can read that file can reuse the credential; treat a shared Brev instance accordingly."
    else
        echo "[github-auth] Credential source: persistent GitHub CLI login (secure-store details are managed by gh)."
    fi
fi

repository_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail \
    "Run this script from inside the repository clone."
remote_url="$(git -C "$repository_root" remote get-url "$GIT_REMOTE" 2>/dev/null)" || fail \
    "The repository has no '$GIT_REMOTE' remote."
repository="$(resolve_github_repository "$remote_url")" || fail \
    "The '$GIT_REMOTE' remote is not a supported github.com repository URL."

push_permission="$(gh api --hostname "$GITHUB_HOST" "repos/$repository" --jq '.permissions.push // false' 2>/dev/null)" || fail \
    "Account '$active_account' cannot inspect push permission for $repository."
if [[ "$push_permission" != "true" ]]; then
    fail "Account '$active_account' does not have push permission to $repository. Ask a maintainer for access or log in with a different account."
fi
echo "[github-auth] Verified: $active_account has GitHub API push permission to $repository."

gh auth setup-git --hostname "$GITHUB_HOST"
gh api --hostname "$GITHUB_HOST" user --jq '.login' >/dev/null 2>&1 || fail \
    "GitHub CLI authentication became unavailable while configuring Git."

echo "[github-auth] GitHub CLI and the HTTPS Git credential helper are ready for $repository."
echo "[github-auth] Do not leave a personal login on a Brev instance shared with untrusted users."
if [[ "$auth_source" == "environment" ]]; then
    echo "[github-auth] Cleanup for this transient login: unset GH_TOKEN GITHUB_TOKEN"
    echo "[github-auth] If a stored login is still active afterward, log it out separately with gh auth logout."
else
    echo "[github-auth] Cleanup when finished: gh auth logout --hostname $GITHUB_HOST --user $active_account"
    echo "[github-auth] Then verify removal without displaying a token: gh auth status --hostname $GITHUB_HOST >/dev/null 2>&1"
fi
