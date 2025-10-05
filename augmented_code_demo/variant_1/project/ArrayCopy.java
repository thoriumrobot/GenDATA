    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ArrayCopy {

    @Positive
  void copy(int @MinLen(1) [] nums) {
    @Positive
    int[] nums_copy = new int[nums.length];
    @Positive
    System.arraycopy(nums, 0, nums_copy, 0, nums.length);
    @Positive
    nums = nums_copy;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
