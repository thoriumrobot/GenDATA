// Source-based slice around line 436
// Method: <com.google.common.math.BigIntegerMath: BigInteger listProduct(List)>

    }
    // Check for leftovers.
    if (product > 1) {
      bignums.add(BigInteger.valueOf(product));
    }
    // Efficiently multiply all the intermediate products together.
    return listProduct(bignums).shiftLeft(shift);
  }

  static BigInteger listProduct(List<BigInteger> nums) {
    return listProduct(nums, 0, nums.size());
  }

  static BigInteger listProduct(List<BigInteger> nums, int start, int end) {
    switch (end - start) {
      case 0:
        return BigInteger.ONE;
      case 1:
        return nums.get(start);
      case 2:
