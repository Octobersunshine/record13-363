import warnings
import numpy as np
from quantile_service import (
    QuantileService,
    compute_quantile,
    compute_quantile_batch,
    compute_quantile_multi_method,
    compute_five_number_summary,
)


def test_basic_quantiles():
    print("=" * 60)
    print("测试1: 基本分位数计算 (数据: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])")
    print("=" * 60)

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    qs = QuantileService()
    quantiles = [0.25, 0.5, 0.75]

    for method in ["linear", "midpoint", "nearest"]:
        result = qs.compute(data, quantiles, method=method)
        np_result = np.percentile(data, [q * 100 for q in quantiles], method=method)
        print(f"\n方法: {method}")
        print(f"  我们的实现: {result}")
        print(f"  NumPy 结果: {np_result}")
        assert np.allclose(result, np_result), f"{method} 方法结果不匹配!"

    print("\n✓ 基本分位数计算通过")


def test_single_quantile():
    print("\n" + "=" * 60)
    print("测试2: 单分位点计算")
    print("=" * 60)

    data = [10, 20, 30, 40, 50]
    qs = QuantileService()

    result_linear = qs.compute(data, 0.5, method="linear")
    np_result = np.percentile(data, 50, method="linear")
    print(f"中位数 (线性): {result_linear}, NumPy: {np_result}")
    assert result_linear == np_result

    result_fn = compute_quantile(data, 0.5, method="linear")
    assert result_fn == result_linear

    print("\n✓ 单分位点计算通过")


def test_edge_cases():
    print("\n" + "=" * 60)
    print("测试3: 边界情况 (q=0, q=1)")
    print("=" * 60)

    data = [3, 1, 4, 1, 5, 9, 2, 6]
    qs = QuantileService()

    for method in ["linear", "midpoint", "nearest"]:
        q0 = qs.compute(data, 0.0, method=method)
        q1 = qs.compute(data, 1.0, method=method)
        np_q0 = np.percentile(data, 0, method=method)
        np_q1 = np.percentile(data, 100, method=method)
        print(f"方法 {method}: q0={q0}, q1={q1}, NumPy q0={np_q0}, q1={np_q1}")
        assert q0 == np_q0 and q1 == np_q1

    print("\n✓ 边界情况通过")


def test_input_validation():
    print("\n" + "=" * 60)
    print("测试4: 输入验证")
    print("=" * 60)

    qs = QuantileService()
    valid_data = [1, 2, 3, 4, 5]

    try:
        qs.compute([], 0.5)
        print("ERROR: 应该对空数据抛出异常")
        assert False
    except ValueError as e:
        print(f"✓ 空数据检测: {e}")

    try:
        qs.compute(valid_data, -0.1)
        print("ERROR: 应该对负分位点抛出异常")
        assert False
    except ValueError as e:
        print(f"✓ 负分位点检测: {e}")

    try:
        qs.compute(valid_data, 1.5)
        print("ERROR: 应该对超范围分位点抛出异常")
        assert False
    except ValueError as e:
        print(f"✓ 超范围分位点检测: {e}")

    try:
        qs.compute(valid_data, 0.5, method="invalid")
        print("ERROR: 应该对无效方法抛出异常")
        assert False
    except ValueError as e:
        print(f"✓ 无效方法检测: {e}")

    print("\n✓ 输入验证通过")


def test_numpy_array_input():
    print("\n" + "=" * 60)
    print("测试5: NumPy 数组输入")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.randn(100)
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    qs = QuantileService()

    for method in ["linear", "midpoint", "nearest"]:
        result = qs.compute(data, quantiles, method=method)
        np_result = np.percentile(data, quantiles * 100, method=method)
        print(f"方法 {method}: 结果匹配 = {np.allclose(result, np_result)}")
        assert np.allclose(result, np_result)

    print("\n✓ NumPy 数组输入通过")


def test_interpolation_demonstration():
    print("\n" + "=" * 60)
    print("测试6: 三种插值方法差异演示")
    print("=" * 60)

    data = [1, 3, 5, 7, 9]
    print(f"数据: {data}")
    print(f"排序后: {sorted(data)}")
    print(f"计算 q=0.6 (位置 = 0.6 * 4 = 2.4):")

    qs = QuantileService()
    for method in ["linear", "midpoint", "nearest"]:
        result = qs.compute(data, 0.6, method=method)
        print(f"  {method:>8s}: {result}")

    print(f"\n说明:")
    print(f"  位置 2.4 介于索引 2 (值=5) 和索引 3 (值=7) 之间")
    print(f"  linear: 5 + 0.4*(7-5) = 5.8")
    print(f"  midpoint: (5+7)/2 = 6.0")
    print(f"  nearest: round(2.4)=2, 值=5")


def test_small_sample():
    print("\n" + "=" * 60)
    print("测试7: 小样本 (n < 2) 分位数计算")
    print("=" * 60)

    qs = QuantileService()

    print("\n--- n=1: 单元素数据 ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = qs.compute([42], 0.5, method="linear")
        assert len(w) == 1
        assert "Sample size is 1" in str(w[0].message)
        assert result == 42.0
        print(f"  单分位点 q=0.5: {result} (预期 42.0) ✓")
        print(f"  警告信息: {w[0].message} ✓")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = qs.compute([42], [0.25, 0.5, 0.75], method="linear")
        assert len(w) == 1
        assert np.all(results == 42.0)
        print(f"  多分位点 [0.25,0.5,0.75]: {results} (预期全为 42.0) ✓")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = qs.compute([7], 0.5, method="midpoint")
        assert len(w) == 1
        assert result == 7.0
        print(f"  midpoint 方法 q=0.5: {result} (预期 7.0) ✓")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = qs.compute([7], 0.5, method="nearest")
        assert len(w) == 1
        assert result == 7.0
        print(f"  nearest 方法 q=0.5: {result} (预期 7.0) ✓")

    print("\n✓ 小样本测试通过")


def test_compute_batch():
    print("\n" + "=" * 60)
    print("测试8: compute_batch - 返回分位点-值字典映射")
    print("=" * 60)

    qs = QuantileService()
    data = list(range(1, 11))
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    result = qs.compute_batch(data, quantiles, method="linear")
    print(f"\n输入数据: {data}")
    print(f"分位点列表: {quantiles}")
    print("结果 (linear):")
    for q, v in result.items():
        np_v = np.percentile(data, q * 100, method="linear")
        print(f"  q={q:<4g}: {v:<6.3f}  (NumPy: {np_v:<6.3f})")
        assert abs(v - np_v) < 1e-10

    assert set(result.keys()) == set(float(q) for q in quantiles)

    try:
        qs.compute_batch(data, 0.5)
        assert False, "应该对单个分位点抛出 TypeError"
    except TypeError as e:
        print(f"\n✓ 单分位点类型检测: {e}")

    fn_result = compute_quantile_batch(data, quantiles, method="midpoint")
    for q, v in fn_result.items():
        np_v = np.percentile(data, q * 100, method="midpoint")
        assert abs(v - np_v) < 1e-10
    print("✓ compute_quantile_batch 便捷函数通过")

    print("\n✓ compute_batch 测试通过")


def test_compute_multi_method():
    print("\n" + "=" * 60)
    print("测试9: compute_multi_method - 多种插值方法同时计算")
    print("=" * 60)

    qs = QuantileService()
    data = list(range(1, 11))
    quantiles = [0.25, 0.5, 0.75]

    result = qs.compute_multi_method(data, quantiles)
    print(f"\n输入数据: {data}")
    print(f"分位点: {quantiles}")
    for method in ["linear", "midpoint", "nearest"]:
        np_v = np.percentile(data, [q * 100 for q in quantiles], method=method)
        our_v = result[method]
        print(f"\n{method}:")
        print(f"  我们:    {our_v}")
        print(f"  NumPy:  {np_v}")
        assert np.allclose(our_v, np_v)
    assert set(result.keys()) == {"linear", "midpoint", "nearest"}

    subset = qs.compute_multi_method(data, 0.5, methods=["linear", "nearest"])
    print(f"\n子集方法 q=0.5: linear={subset['linear']}, nearest={subset['nearest']}")
    assert subset["linear"] == 5.5
    assert subset["nearest"] == 5.0

    try:
        qs.compute_multi_method(data, 0.5, methods=["linear", "invalid"])
        assert False, "应该对无效方法抛出 ValueError"
    except ValueError as e:
        print(f"\n✓ 无效方法检测: {e}")

    fn_result = compute_quantile_multi_method(data, quantiles)
    for m in ["linear", "midpoint", "nearest"]:
        assert np.allclose(fn_result[m], result[m])
    print("✓ compute_quantile_multi_method 便捷函数通过")

    print("\n✓ compute_multi_method 测试通过")


def test_five_number_summary():
    print("\n" + "=" * 60)
    print("测试10: compute_five_number_summary - 五数概括")
    print("=" * 60)

    qs = QuantileService()
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    sorted_data = sorted(data)
    print(f"\n输入数据: {data}")
    print(f"排序后:  {sorted_data}")

    for method in ["linear", "midpoint", "nearest"]:
        result = qs.compute_five_number_summary(data, method=method)
        print(f"\n方法 {method}:")
        for label, v in result.items():
            print(f"  {label:<7s}: {v}")
        np_min = np.percentile(data, 0, method=method)
        np_q1 = np.percentile(data, 25, method=method)
        np_med = np.percentile(data, 50, method=method)
        np_q3 = np.percentile(data, 75, method=method)
        np_max = np.percentile(data, 100, method=method)
        assert result["min"] == np_min
        assert abs(result["q1"] - np_q1) < 1e-10
        assert abs(result["median"] - np_med) < 1e-10
        assert abs(result["q3"] - np_q3) < 1e-10
        assert result["max"] == np_max

    fn_result = compute_five_number_summary(data, method="linear")
    class_result = qs.compute_five_number_summary(data, method="linear")
    assert fn_result == class_result
    print("\n✓ compute_five_number_summary 便捷函数通过")

    print("\n✓ 五数概括测试通过")


if __name__ == "__main__":
    test_basic_quantiles()
    test_single_quantile()
    test_edge_cases()
    test_input_validation()
    test_numpy_array_input()
    test_interpolation_demonstration()
    test_small_sample()
    test_compute_batch()
    test_compute_multi_method()
    test_five_number_summary()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
