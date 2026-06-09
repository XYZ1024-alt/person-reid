# Project Architecture

![PedestrianDetection Architecture](architecture.png)

```mermaid
flowchart LR
    subgraph data["Data Sources"]
        market["Market-1501<br/>train / query / gallery"]
        prcc["PRCC<br/>rgb + optional sketch"]
        target["data/target.jpg"]
        video["data/video.mp4"]
    end

    subgraph cli["Command Entry Points"]
        train_cli["python -m scripts.train"]
        eval_cli["python -m scripts.evaluate"]
        extract_cli["python -m scripts.extract"]
        plot_cli["python -m scripts.plot_metrics"]
        demo_cli["python main.py"]
        run_script["run.sh<br/>5-stage transfer recipe"]
    end

    subgraph reid_data["pedestrian_reid.data"]
        datasets["datasets.py<br/>sample loading + labels"]
        transforms["transforms.py<br/>train/eval augmentation"]
        samplers["samplers.py<br/>identity / clothes-aware / source-balanced batches"]
    end

    subgraph reid_core["pedestrian_reid core"]
        builders["builders.py<br/>datasets + dataloaders"]
        model["modules/model.py<br/>ResNet50-IBN + BNNeck + CAL head"]
        losses["modules/losses.py<br/>triplet loss"]
        metrics["modules/metrics.py<br/>feature banks + ReID metrics"]
        trainer["engine/trainer.py<br/>training loop + checkpointing"]
        evaluator["engine/evaluator.py<br/>standard / dark / occluded validation"]
        runtime["runtime.py<br/>torch runtime setup"]
    end

    subgraph artifacts["Output Artifacts"]
        checkpoints["outputs/.../best.pth<br/>outputs/.../last.pth"]
        run_config["run_config.json"]
        train_csv["training_metrics.csv"]
        eval_csv["evaluation_metrics.csv"]
        figures["outputs/.../figures"]
    end

    subgraph inference["Detection + ReID Demo"]
        loader["models/loader.py<br/>YOLO + ReID predictor factories"]
        detector["modules/detector.py<br/>person detection + crops"]
        reid_engine["modules/reid_engine.py<br/>feature extraction + cosine similarity"]
        overlay["OpenCV display<br/>target match boxes"]
    end

    subgraph external["External Runtime Dependencies"]
        torch["PyTorch / torchvision"]
        yolo["Ultralytics YOLO"]
        cv2["OpenCV"]
        pil["Pillow"]
    end

    run_script --> train_cli
    train_cli --> runtime
    train_cli --> trainer
    trainer --> builders
    builders --> datasets
    builders --> transforms
    builders --> samplers
    market --> datasets
    prcc --> datasets
    trainer --> model
    trainer --> losses
    trainer --> evaluator
    evaluator --> builders
    evaluator --> metrics
    model --> checkpoints
    trainer --> checkpoints
    trainer --> run_config
    trainer --> train_csv
    evaluator --> eval_csv
    eval_cli --> runtime
    eval_cli --> evaluator
    eval_cli --> checkpoints
    extract_cli --> checkpoints
    plot_cli --> train_csv
    plot_cli --> eval_csv
    plot_cli --> figures

    demo_cli --> loader
    loader --> yolo
    loader --> checkpoints
    loader --> model
    loader --> transforms
    target --> detector
    video --> detector
    detector --> reid_engine
    loader --> reid_engine
    reid_engine --> overlay
    demo_cli --> overlay

    torch --> model
    torch --> trainer
    torch --> evaluator
    yolo --> detector
    cv2 --> demo_cli
    pil --> loader
```

## Legend

- Training path: `scripts.train` builds datasets/loaders, trains `PedestrianReIDNet`, evaluates periodically, and writes checkpoints plus metrics.
- Evaluation path: `scripts.evaluate` loads a checkpoint and reports standard, dark-query, and occluded-query retrieval metrics.
- Demo path: `main.py` detects people with YOLO, extracts ReID features from each crop, compares them with the target image by cosine similarity, and renders matched boxes with OpenCV.
