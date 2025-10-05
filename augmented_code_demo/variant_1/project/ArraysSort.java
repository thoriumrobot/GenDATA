    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ArraysSort {

    @Positive
  void sortInt(int @MinLen(10) [] nums) {
    // Checks the correct handling of the toIndex parameter
    @Positive
    Arrays.sort(nums, 0, 10);
    // :: error: (argument)
    @Positive
    Arrays.sort(nums, 0, 11);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
