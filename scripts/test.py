# --- NEW: helpers to recognize & build modules from structureDB entries ---
import ast
from collections import OrderedDict
import bcs
import multiprocessing as mp
def is_extension_module_entry(val):
    """
    Heuristic:
    - A module spec often appears as list/tuple/dict describing domains (strings like "AT{...}", "KR{...}", "loading: ...")
    - Exclude RDKit Mol (has GetNumAtoms), plain strings, ints, etc.
    - Include existing bcs.Module instances (clone later).
    """
    # RDKit Mol check (common in structureDB)
    if hasattr(val, "GetNumAtoms"):  # RDKit Mol interface
        return False
    if isinstance(val, bcs.Module):
        return True
    if isinstance(val, (list, tuple, dict)):
        return True
    return False

def spec_to_module(spec):
    """
    Turn a structureDB module spec like:
        ["AT{'substrate': 'D-isobutmal'}", "KR{'type': 'C2'}", 'loading: False']
    into a real bcs.Module with proper domain instances.
    """
    loading = False
    domains = OrderedDict()

    if isinstance(spec, dict):
        # If your structureDB sometimes uses dicts like {"domains": [...], "loading": False}
        if "loading" in spec:
            loading = bool(spec["loading"])
        spec_list = spec.get("domains", [])
    elif isinstance(spec, (list, tuple)):
        spec_list = spec
    else:
        raise TypeError(f"Unsupported module spec type: {type(spec)}")

    for item in spec_list:
        if not isinstance(item, str):
            # Already a constructed domain object? If so, attach directly.
            # (Adjust if your bcs domains carry a 'cls' attribute you can use as key.)
            if hasattr(item, "__class__") and item.__class__.__name__ in ("AT", "KS", "KR", "DH", "ER", "ACP", "TE"):
                domain_cls = getattr(bcs, item.__class__.__name__)
                domains[domain_cls] = item
            continue

        s = item.strip()
        # loading flag line like 'loading: False'
        if s.lower().startswith("loading"):
            try:
                _, val = s.split(":", 1)
                loading = val.strip().lower() == "true"
            except Exception:
                pass
            continue

        # lines like "AT{'substrate': 'D-isobutmal'}"
        if "{" in s and s.endswith("}"):
            name, cfg = s.split("{", 1)
            name = name.strip()
            cfg_dict = ast.literal_eval("{" + cfg)  # safe because it’s only dict-literals
        else:
            # bare domain name like "ACP" with no args
            name, cfg_dict = s, {}

        if not hasattr(bcs, name):
            raise ValueError(f"Unknown bcs domain '{name}' in spec: {s}")

        domain_cls = getattr(bcs, name)
        domain_obj = domain_cls(**cfg_dict) if cfg_dict else domain_cls()
        domains[domain_cls] = domain_obj

    return bcs.Module(domains=domains, loading=loading)

# --- UPDATED worker: never call structureDB entries; instantiate appropriately ---
def build_bcs_cluster_and_product(args):
    starter_code, extension_keys = args
    from retrotide import structureDB
    import bcs as _bcs
    from collections import OrderedDict as _OD

    try:
        # loading module
        loading_AT = _bcs.AT(active=True, substrate=starter_code)
        loading_module = _bcs.Module(domains=_OD({_bcs.AT: loading_AT}), loading=True)

        # build each extension module from its structureDB entry
        ext_modules = []
        for key in extension_keys:
            entry = structureDB[key]
            if isinstance(entry, _bcs.Module):
                # clone defensively (avoid sharing instances across workers)
                # If bcs.Module has a copy/clone, use it; else rebuild domains:
                cloned = spec_to_module(
                    {"domains": [f"{d.__class__.__name__}{repr(getattr(d, '__dict__', {}))}" for d in entry.domains.values()],
                     "loading": getattr(entry, "loading", False)}
                )
                ext_modules.append(cloned)
            elif is_extension_module_entry(entry):
                ext_modules.append(spec_to_module(entry))
            else:
                # Not a module spec -> skip this combo
                return ("ERROR", starter_code, tuple(extension_keys),
                        f"Non-module entry for key '{key}' (type: {type(entry).__name__})")

        cluster = _bcs.Cluster(modules=[loading_module] + ext_modules)
        from retrotide import structureDB as _sdb
        product_mol = cluster.computeProduct(_sdb)
        return cluster, product_mol

    except Exception as e:
        return ("ERROR", starter_code, tuple(extension_keys), repr(e))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Choose allowed starters/extenders (don’t mutate bcs.starters/extenders)
    starter_codes = None  # allow all starters
    extender_codes = ['Malonyl-CoA', 'Methylmalonyl-CoA']  # restrict extenders
    starters, extenders = filter_units(starter_codes, extender_codes)

    print(f"\nNumber of starter units: {len(starters)}")
    print(f"Number of extender units: {len(extenders)}\n")

    # Import structureDB only after filters are decided (children import separately)
    from retrotide import retrotide, structureDB  # noqa: F401
    print(f"\nNumber of entries in structureDB: {len(structureDB)}\n")

    # IMPORTANT: only include structureDB keys that look like module specs (not RDKit Mols)
    extension_module_keys = [k for k, v in structureDB.items() if is_extension_module_entry(v)]
    starter_keys = list(starters.keys())

    # --- NEW: print how many clusters will get generated (by extension length and total) ---
    num_starters = len(starter_keys)
    num_extmods = len(extension_module_keys)
    per_i_counts = [num_starters * (num_extmods ** i) for i in range(1, MAX_EXTENSION_MODULES + 1)]
    total_combos = sum(per_i_counts)

    print("Planned generation workload:")
    for i, count in enumerate(per_i_counts, start=1):
        print(f"  - {count:,} combinations with {i} extension module(s)")
    print(f"Total combinations to attempt: {total_combos:,}\n")
    # --- END NEW ---

    all_cluster_product_pairs = []
    errors = []

    processes = max(1, mp.cpu_count() - 1)
    chunksize = 64

    with mp.get_context("spawn").Pool(processes=processes, maxtasksperchild=500) as pool:
        # keep streaming generator — no need to materialize the args list
        args_iter = generate_arg_stream(starter_keys, extension_module_keys, MAX_EXTENSION_MODULES)
        for res in pool.imap_unordered(build_bcs_cluster_and_product, args_iter, chunksize=chunksize):
            if isinstance(res, tuple) and len(res) > 0 and res[0] == "ERROR":
                errors.append(res)
            else:
                all_cluster_product_pairs.append(res)

    print(f"Successfully generated {len(all_cluster_product_pairs)} (cluster, product) pairs.")
    if errors:
        print(f"{len(errors)} combinations failed; first few errors:\n" +
              "\n".join(map(str, errors[:5])))
