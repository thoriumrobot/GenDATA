// Source-based slice around line 440
// Method: <com.google.common.math.BigIntegerMath: BigInteger listProduct(List,int,int)>

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
        return nums.get(start).multiply(nums.get(start + 1));
      case 3:
        return nums.get(start).multiply(nums.get(start + 1)).multiply(nums.get(start + 2));
      default:
