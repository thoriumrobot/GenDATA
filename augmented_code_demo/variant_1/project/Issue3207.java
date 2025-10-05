// Test case for https://tinyurl.com/cfissue/3207

    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class Issue3207 {

    @Positive
  void m(int @MinLen(1) [] arr) {
    @Positive
    @LTLengthOf("arr") int j = 0;
    @Positive
  }

    @Positive
  void m2(int @MinLen(1) [] @MinLen(1) [] arr) {
    @Positive
    @LTLengthOf("arr[0]") int j = 0;
    @Positive
  }

    @Positive
  void m3(int @MinLen(1) [] @MinLen(1) [] arr) {
    @Positive
    int @MinLen(1) [] arr0 = arr[0];
    @Positive
    @LTLengthOf("arr0") int j = 0;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
