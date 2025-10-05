    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;

    @Positive
public class ArrayLength {
    @Positive
  void test() {
    @Positive
    int[] arr = {1, 2, 3};
    @Positive
    @LTEqLengthOf({"arr"}) int a = arr.length;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
