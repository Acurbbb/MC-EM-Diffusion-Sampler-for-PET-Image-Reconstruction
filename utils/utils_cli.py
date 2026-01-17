from typing import List
from omegaconf import OmegaConf

def _normalize_overrides(unknown: List[str]) -> List[str]:
    """
    Turn things like:
      ['--a.b', '1', 'c.d=2', 'e.f', 'hello', '---weird', '--x.y=3']
    into:
      ['a.b=1', 'c.d=2', 'e.f=hello', 'x.y=3']
    Rules:
      - Strip leading dashes.
      - If a token has '=', keep it.
      - Else, pair it with the next token if present and not a new option.
    """
    out = []
    i = 0
    while i < len(unknown):
        tok = unknown[i].lstrip('-')  # strip any number of leading '-'
        if '=' in tok:
            out.append(tok)
            i += 1
            continue
        # no '=' — try to pair with the next token
        if i + 1 < len(unknown):
            nxt = unknown[i + 1]
            # if the next token starts with '-' it's probably another key; treat current as a flag without value
            if nxt.startswith('-'):
                # interpret as boolean True
                out.append(f"{tok}=true")
                i += 1
            else:
                out.append(f"{tok}={nxt}")
                i += 2
        else:
            # dangling key -> boolean True
            out.append(f"{tok}=true")
            i += 1
    return out

def _to_plain(cfg): 
    return OmegaConf.to_container(cfg, resolve=True)

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def diff_cfg(base: OmegaConf, new: OmegaConf):
    fb = _flatten(_to_plain(base))
    fn = _flatten(_to_plain(new))
    paths = sorted(set(fb) | set(fn))
    return [(p, fb.get(p, "<MISSING>"), fn.get(p, "<MISSING>"))
            for p in paths if fb.get(p) != fn.get(p)]

def print_config_report(yaml_conf: OmegaConf, merged_cfg: OmegaConf) -> None:
    """Print applied overrides relative to YAML base, and the final merged config."""
    changes = diff_cfg(yaml_conf, merged_cfg)
    if changes:
        print("\n=== Applied overrides (old -> new) ===")
        for path, old, new in changes:
            print(f"{path}: {old} -> {new}")
    else:
        print("\n(No changes from base YAML)")

    print("\n=== Final merged config ===")
    print(OmegaConf.to_yaml(merged_cfg, resolve=True))