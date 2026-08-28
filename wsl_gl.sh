# wsl_gl.sh - pick the OpenGL / matplotlib setup for running the ACT sims under WSL.
#
# MuJoCo/dm_control choose their GL backend at import time from $MUJOCO_GL, and
# under WSL the default (GLX/GLFW) path routinely aborts with
#   xcb_io.c: ... _XReply ... Aborting
# because it wants a real X display. This repo always renders offscreen
# (physics.render() -> ndarray), so EGL (GPU, headless) is the reliable choice
# in BOTH modes below. The only thing an X server buys you here is the live
# matplotlib preview window that --onscreen_render pops up.
#
# Usage - source it, then run scripts normally:
#
#   source wsl_gl.sh headless    # offscreen only: EGL + matplotlib Agg
#                                # (record data, train, write eval videos)
#   source wsl_gl.sh onscreen    # also show the live preview via WSLg X:
#                                # EGL + matplotlib TkAgg + $DISPLAY
#   source wsl_gl.sh status      # show current settings + run a render self-test
#
#   uv run python imitate_episodes.py ...
#
# Or as a one-off wrapper (no sourcing needed):
#
#   ./wsl_gl.sh headless uv run python record_sim_episodes.py --task_name ...

_wgl_mode="${1:-status}"
[ "$#" -gt 0 ] && shift

# --- are we being sourced? ---------------------------------------------------
_wgl_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in *:file) _wgl_sourced=1 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
    (return 0 2>/dev/null) && _wgl_sourced=1
fi
_wgl_finish() {  # $1 = exit code; return if sourced, exit otherwise
    unset -f _wgl_finish _wgl_is_wsl _wgl_apply _wgl_selftest 2>/dev/null
    if [ "$_wgl_sourced" = 1 ]; then return "${1:-0}"; else exit "${1:-0}"; fi
}

_wgl_is_wsl() {
    [ -n "${WSL_DISTRO_NAME:-}" ] || [ -n "${WSL_INTEROP:-}" ] && return 0
    grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null
}

_wgl_apply() {
    case "$_wgl_mode" in
        headless)
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
            unset DISPLAY            # nothing should touch X in this mode
            echo "[wsl_gl] headless: MUJOCO_GL=egl  MPLBACKEND=Agg  DISPLAY unset"
            ;;
        onscreen|x11|x)
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=TkAgg
            export DISPLAY="${DISPLAY:-:0}"
            if command -v xset >/dev/null 2>&1 && xset q >/dev/null 2>&1; then
                echo "[wsl_gl] onscreen: MUJOCO_GL=egl  MPLBACKEND=TkAgg  DISPLAY=$DISPLAY (X reachable)"
            else
                echo "[wsl_gl] onscreen: MUJOCO_GL=egl  MPLBACKEND=TkAgg  DISPLAY=$DISPLAY"
                echo "[wsl_gl] WARNING: cannot reach X at $DISPLAY - is WSLg up? run 'wsl --update' on Windows."
            fi
            ;;
        status) : ;;
        *)
            echo "[wsl_gl] unknown mode '$_wgl_mode' (use: headless | onscreen | status)" >&2
            return 2
            ;;
    esac
}

_wgl_selftest() {
    echo "[wsl_gl] current: MUJOCO_GL=${MUJOCO_GL:-<unset>}  PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>}  MPLBACKEND=${MPLBACKEND:-<unset>}  DISPLAY=${DISPLAY:-<unset>}"
    if [ -z "${MUJOCO_GL:-}" ]; then
        echo "[wsl_gl] MUJOCO_GL is not set - run 'source wsl_gl.sh headless' (or onscreen) first."
        echo "[wsl_gl] skipping render self-test (it would use the default GLX path and can abort under WSL)."
        return 0
    fi
    command -v uv >/dev/null 2>&1 || { echo "[wsl_gl] uv not found - skipping render self-test"; return 0; }
    uv run python - <<'PY'
import os, sys
try:
    from dm_control import mujoco
    xml = ("<mujoco><worldbody><light pos='0 0 1'/>"
           "<geom type='box' size='.1 .1 .1' rgba='.7 .3 .3 1'/></worldbody></mujoco>")
    img = mujoco.Physics.from_xml_string(xml).render(height=64, width=64)
    print(f"[wsl_gl] offscreen render OK -> {img.shape}, backend={os.environ.get('MUJOCO_GL')}")
except Exception as exc:  # noqa: BLE001
    print(f"[wsl_gl] offscreen render FAILED: {exc!r}", file=sys.stderr)
    sys.exit(1)
PY
}

if _wgl_is_wsl; then
    _wgl_apply || _wgl_finish $?
else
    echo "[wsl_gl] not running under WSL - no changes made."
fi

if [ "$#" -gt 0 ]; then
    "$@"; _wgl_finish $?
elif [ "$_wgl_mode" = status ]; then
    _wgl_selftest; _wgl_finish $?
elif [ "$_wgl_sourced" != 1 ]; then
    echo "[wsl_gl] tip: 'source wsl_gl.sh $_wgl_mode' to keep the settings, or"
    echo "[wsl_gl]      './wsl_gl.sh $_wgl_mode <command...>' to wrap one command."
fi
_wgl_finish 0
