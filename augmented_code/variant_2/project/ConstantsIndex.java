/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ConstantsIndex {

    @Positive
  void test() {
    @Positive
    int @MinLen(3) [] arr = {1, 2, 3};
    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class ConstantsIndex {

    @Positive
  void test() {
    @Positive
    int @MinLen(3) [] arr = {1, 2, 3};
    @Positive
    int arr[1] = arr[1];
    // :: error: (array.access.unsafe.high.constant)
    @Positive
    int j = arr[3];
    @Positive
  }
    @Positive
}

    // :: error: (array.access.unsafe.high.constant)
    @Positive
    /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    int arr[3] = arr[3];
    @Positive
  }
    @Positive
}

    @Positive
  }
    @Positive
}
