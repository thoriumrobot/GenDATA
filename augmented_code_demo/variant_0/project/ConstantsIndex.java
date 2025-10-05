    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ConstantsIndex {

    @Positive
  void test() {
    @Positive
    int @MinLen(3) [] arr = {1, 2, 3};
    @Positive
    int i = arr[1];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    int j = arr[3];
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
