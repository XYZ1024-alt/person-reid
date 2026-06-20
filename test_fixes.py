#!/usr/bin/env python3
"""验证PRCC修复的测试脚本"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from pedestrian_reid.engine.trainer import _classification_losses, _prcc_local_labels
from pedestrian_reid.data.datasets import PRCC_SOURCE, MARKET_SOURCE


def test_prcc_local_labels():
    """测试PRCC标签偏移逻辑"""
    print("Testing _prcc_local_labels()...")

    # Test 1: Joint mode (both Market and PRCC samples)
    labels = torch.tensor([10, 20, 751, 800, 900])  # Market: [10,20], PRCC: [751,800,900]
    prcc_mask = torch.tensor([False, False, True, True, True])
    num_market_classes = 751

    result = _prcc_local_labels(labels, prcc_mask, num_market_classes)
    expected = torch.tensor([10, 20, 0, 49, 149])  # PRCC labels shifted to [0, num_prcc_classes)

    assert torch.equal(result, expected), f"Joint mode failed: {result} != {expected}"
    print("  ✓ Joint mode: PRCC labels correctly shifted")

    # Test 2: Pure PRCC mode (no Market samples)
    labels = torch.tensor([0, 50, 100, 150, 199])  # All PRCC in local range
    prcc_mask = torch.tensor([True, True, True, True, True])

    result = _prcc_local_labels(labels, prcc_mask, num_market_classes)
    expected = labels.clone()  # Should NOT shift

    assert torch.equal(result, expected), f"Pure PRCC mode failed: {result} != {expected}"
    print("  ✓ Pure PRCC mode: labels NOT shifted (already in local range)")

    # Test 3: Pure Market mode (no PRCC samples)
    labels = torch.tensor([0, 100, 200, 500, 750])
    prcc_mask = torch.tensor([False, False, False, False, False])

    result = _prcc_local_labels(labels, prcc_mask, num_market_classes)
    expected = labels.clone()  # Should NOT shift

    assert torch.equal(result, expected), f"Pure Market mode failed: {result} != {expected}"
    print("  ✓ Pure Market mode: labels NOT shifted")

    # Test 4: num_market_classes = 0 (single classifier mode)
    labels = torch.tensor([0, 50, 100])
    prcc_mask = torch.tensor([True, True, True])
    num_market_classes = 0

    result = _prcc_local_labels(labels, prcc_mask, num_market_classes)
    expected = labels.clone()

    assert torch.equal(result, expected), f"Single classifier mode failed: {result} != {expected}"
    print("  ✓ Single classifier mode: labels NOT shifted")

    print("✅ All _prcc_local_labels tests passed!\n")


def test_classification_losses():
    """测试分类损失函数"""
    print("Testing _classification_losses()...")

    device = torch.device("cpu")
    batch_size = 4
    num_market_classes = 751
    num_prcc_classes = 200

    # Test 1: Dual classifier, pure PRCC mode
    outputs = {
        "logits": torch.randn(batch_size, num_market_classes),  # Market head
        "prcc_logits": torch.randn(batch_size, num_prcc_classes),  # PRCC head
    }
    labels = torch.tensor([0, 50, 100, 150])  # PRCC labels in local range [0, 199]
    sources = [PRCC_SOURCE] * batch_size

    try:
        losses = _classification_losses(outputs, labels, sources, prcc_weight=1.0, device=device, num_market_classes=num_market_classes)
        print(f"  ✓ Pure PRCC mode with dual classifier: total={losses.total:.4f}, market={losses.market:.4f}, prcc={losses.prcc:.4f}")
        assert losses.market.item() == 0.0, "Market loss should be 0 when no Market samples"
        print("  ✓ Market loss correctly zero when no Market samples")
    except Exception as e:
        print(f"  ✗ Pure PRCC mode failed: {e}")
        raise

    # Test 2: Dual classifier, joint mode
    outputs = {
        "logits": torch.randn(batch_size, num_market_classes),
        "prcc_logits": torch.randn(batch_size, num_prcc_classes),
    }
    labels = torch.tensor([10, 20, 751, 800])  # Market: [10,20], PRCC: [751,800]
    sources = [MARKET_SOURCE, MARKET_SOURCE, PRCC_SOURCE, PRCC_SOURCE]

    try:
        losses = _classification_losses(outputs, labels, sources, prcc_weight=1.0, device=device, num_market_classes=num_market_classes)
        print(f"  ✓ Joint mode with dual classifier: total={losses.total:.4f}, market={losses.market:.4f}, prcc={losses.prcc:.4f}")
        assert losses.market.item() > 0.0, "Market loss should be > 0 when Market samples exist"
        assert losses.prcc.item() > 0.0, "PRCC loss should be > 0 when PRCC samples exist"
        print("  ✓ Both losses computed correctly in joint mode")
    except Exception as e:
        print(f"  ✗ Joint mode failed: {e}")
        raise

    print("✅ All _classification_losses tests passed!\n")


def test_gradient_reversal():
    """测试梯度反转优化"""
    print("Testing gradient reversal optimization...")

    from pedestrian_reid.modules.model import PedestrianReIDNet

    # Create model with both clothes classifier and domain discriminator
    model = PedestrianReIDNet(
        num_classes=751,
        num_clothes_classes=11,
        embedding_dim=256,
        use_domain_adversarial=True,
    )

    x = torch.randn(2, 3, 256, 128)
    outputs = model(x)

    assert "clothes_logits" in outputs, "clothes_logits should be in outputs"
    assert "domain_logits" in outputs, "domain_logits should be in outputs"
    print("  ✓ Both clothes and domain discriminators produce outputs")
    print("✅ Gradient reversal optimization validated!\n")


def main():
    print("=" * 60)
    print("PRCC优化方案 - 修复验证测试")
    print("=" * 60)
    print()

    try:
        test_prcc_local_labels()
        test_classification_losses()
        test_gradient_reversal()

        print("=" * 60)
        print("✅ 所有测试通过！修复验证成功。")
        print("=" * 60)
        print()
        print("下一步:")
        print("1. 运行完整训练: bash run.sh 5")
        print("2. 监控mAP是否开始上升")
        print("3. 检查训练日志中的损失值")
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
