"""Modal app for distributed dataset generation."""


import modal

app = modal.App("gazet-dataset")

VOLUME_MOUNT = "/data"
INTERMEDIATE_MOUNT = "/intermediate"

volume = modal.Volume.from_name("gazet-data", create_if_missing=True)
intermediate_volume = modal.Volume.from_name(
    "gazet-intermediate", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "duckdb>=1.4.4",
        "fastapi>=0.100",
        "pandas>=2.2",
        "pydantic>=2.0",
        "pyarrow>=17.0.0",
        "pyyaml>=6.0",
    )
    .env({"GAZET_DATA_DIR": VOLUME_MOUNT, "PYTHONPATH": "/root"})
    .add_local_dir("src/gazet", "/root/gazet")
    .add_local_dir("dataset", "/root/dataset")
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume, INTERMEDIATE_MOUNT: intermediate_volume},
    timeout=300,
    cpu=2,
    memory=4096,
)
def build_inventory_remote():
    """Build entity inventory from parquet files on the volume."""
    from pathlib import Path
    from dataset.scripts.build_inventory import build_inventory_to_dir

    result = build_inventory_to_dir(Path(INTERMEDIATE_MOUNT))
    intermediate_volume.commit()
    return result


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume, INTERMEDIATE_MOUNT: intermediate_volume},
    timeout=3600,
    cpu=4,
    memory=32768,
)
def build_relation_remote(relation_type: str, countries: list, limit: int):
    """Compute one relation type and save to intermediate volume."""
    from pathlib import Path
    from dataset.scripts.build_relations import compute_single_relation

    count = compute_single_relation(
        relation_type=relation_type,
        countries=countries,
        limit=limit,
        output_dir=Path(INTERMEDIATE_MOUNT),
    )
    intermediate_volume.commit()
    return {"relation_type": relation_type, "count": count}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume, INTERMEDIATE_MOUNT: intermediate_volume},
    timeout=3600,
    cpu=2,
    memory=4096,
)
def generate_batch_remote(work_items: list) -> list:
    """Process a batch of work items on a Modal container."""
    from dataset.scripts.generate_samples import generate_batch_core

    results = generate_batch_core(
        work_items=work_items,
        intermediate_dir=INTERMEDIATE_MOUNT,
    )

    print(f"Batch complete: {sum(1 for r in results if r['sample'])} success / "
          f"{sum(1 for r in results if not r['sample'])} failed out of {len(work_items)}")

    return results


@app.local_entrypoint()
def run_pipeline(
    config_path: str = "dataset/config.yaml",
    num_containers: int = 0,
    skip_inventory: bool = False,
    skip_relations: bool = False,
    fresh: bool = False,
):
    """Run the full distributed pipeline."""
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path(config_path).read_text())
    countries = config["countries"]
    sample_targets = config["sample_targets"]
    modal_cfg = config.get("modal", {})
    n_containers = num_containers or modal_cfg.get("num_containers", 50)
    retry_multiplier = config["generation"]["retry_multiplier"]

    print(f"Countries: {countries}")
    print(f"Sample targets: {sample_targets}")
    print(f"Containers: {n_containers}")

    if not skip_inventory:
        print("Building inventory...")
        result = build_inventory_remote.remote()
        print(f"  Inventory: {result}")

    if not skip_relations:
        print("Building relations...")

        from dataset.scripts.cli import calculate_relation_limits

        relation_needs = calculate_relation_limits(config)

        handles = []
        for rel_type, limit in relation_needs.items():
            h = build_relation_remote.spawn(rel_type, countries, max(limit, 500))
            handles.append((rel_type, h))

        for rel_type, h in handles:
            result = h.get()
            print(f"  {rel_type}: {result['count']} pairs")

    print(f"Generating samples across {n_containers} containers...")

    import json
    from dataset.scripts.generate_samples import prepare_work_items

    output_dir = Path("dataset/output")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "dataset_raw.jsonl"

    existing_samples = []
    sample_counter = 1
    if not fresh and output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    existing_samples.append(json.loads(line))
        if existing_samples:
            max_id = max(
                int(s["id"].split("_")[1])
                for s in existing_samples
                if s["id"].startswith("sample_")
            )
            sample_counter = max_id + 1
            print(f"  Appending to {len(existing_samples)} existing samples")

    work_items = prepare_work_items(
        target_counts=sample_targets,
        retry_multiplier=retry_multiplier,
        start_counter=sample_counter,
        intermediate_dir_str="",
    )

    total_work = len(work_items)
    print(f"  Total work items: {total_work}")

    batch_size = max(1, (total_work + n_containers - 1) // n_containers)
    batches = [
        work_items[i : i + batch_size]
        for i in range(0, total_work, batch_size)
    ]
    print(f"  Batches: {len(batches)} x ~{batch_size} items")

    new_sample_count = 0
    failed_batches = 0
    family_progress = {}

    write_mode = "w" if fresh else "a"
    fout = open(output_file, write_mode)

    try:
        for batch_results in generate_batch_remote.map(
            batches, return_exceptions=True
        ):
            if isinstance(batch_results, Exception):
                failed_batches += 1
                print(f"  Batch failed: {batch_results}")
                continue

            batch_samples = []
            for r in batch_results:
                fam = r["family"]
                if fam not in family_progress:
                    family_progress[fam] = {"success": 0, "failed": 0}
                if r["sample"]:
                    batch_samples.append(r["sample"])
                    family_progress[fam]["success"] += 1
                else:
                    family_progress[fam]["failed"] += 1

            for sample in batch_samples:
                fout.write(json.dumps(sample) + "\n")
            fout.flush()
            new_sample_count += len(batch_samples)

            done = sum(p["success"] + p["failed"] for p in family_progress.values())
            print(f"  Progress: {done}/{total_work} items | {new_sample_count} saved | {failed_batches} batch errors")

    except Exception as e:
        print(f"  Map interrupted: {e}")
    finally:
        fout.close()

    print(f"\nResults by family:")
    for fam in sorted(family_progress.keys()):
        s = family_progress[fam]["success"]
        f = family_progress[fam]["failed"]
        total = s + f
        rate = (s / total * 100) if total > 0 else 0
        target = sample_targets.get(fam, 0)
        print(
            f"  {fam:20s}: {s:4d} success / {f:4d} failed "
            f"({rate:5.1f}%, target: {target})"
        )

    total_samples = len(existing_samples) + new_sample_count
    status = "COMPLETE" if failed_batches == 0 else "PARTIAL"
    print(f"\nGeneration {status}: {new_sample_count} new, {total_samples} total")
    if failed_batches:
        print(f"  Failed batches: {failed_batches}/{len(batches)}")
    print(f"  Output: {output_file}")


@app.local_entrypoint()
def upload_data(data_dir: str = "data"):
    """Upload local data directory to the Modal volume."""
    import os
    from pathlib import Path

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_path} does not exist")
        return

    print(f"Uploading {data_path} to Modal volume 'gazet-data'...")

    file_count = 0
    total_size = 0

    for root, dirs, files in os.walk(data_path):
        for f in files:
            local_path = os.path.join(root, f)
            # Relative path within data_dir becomes the volume path
            rel = os.path.relpath(local_path, data_path)
            size = os.path.getsize(local_path)
            total_size += size
            file_count += 1
            print(f"  {rel} ({size / (1024*1024):.1f} MB)")

    print(f"  {file_count} files, {total_size / (1024*1024):.1f} MB")

    vol = modal.Volume.from_name("gazet-data", create_if_missing=True)
    with vol.batch_upload() as batch:
        batch.put_directory(str(data_path), "/")

    print("Upload complete")
