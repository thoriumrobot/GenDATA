    @Positive
import org.checkerframework.common.value.qual.BottomVal;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class LubIndex {

    @Positive
  public static void MinLen(int @MinLen(10) [] arg, int @MinLen(4) [] arg2) {
    @Positive
    int[] arr;
    @Positive
    if (true) {
    @Positive
      arr = arg;
    @Positive
    } else {
    @Positive
      arr = arg2;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @MinLen(10) [] res = arr;
    @Positive
    int @MinLen(4) [] res2 = arr;
    // :: error: (assignment)
    @Positive
    int @BottomVal [] res3 = arr;
    @Positive
  }

    @Positive
  public static void Bottom(int @BottomVal [] arg, int @MinLen(4) [] arg2) {
    @Positive
    int[] arr;
    @Positive
    if (true) {
    @Positive
      arr = arg;
    @Positive
    } else {
    @Positive
      arr = arg2;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @MinLen(10) [] res = arr;
    @Positive
    int @MinLen(4) [] res2 = arr;
    // :: error: (assignment)
    @Positive
    int @BottomVal [] res3 = arr;
    @Positive
  }

    @Positive
  public static void BothBottom(int @BottomVal [] arg, int @BottomVal [] arg2) {
    @Positive
    int[] arr;
    @Positive
    if (true) {
    @Positive
      arr = arg;
    @Positive
    } else {
    @Positive
      arr = arg2;
    @Positive
    }
    @Positive
    int @MinLen(10) [] res = arr;
    @Positive
    int @BottomVal [] res2 = arr;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
