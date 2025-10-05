    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class ConstantOffsets {
    @Positive
  void method1(int[] a, int offset, @LTLengthOf(value = "#1", offset = "-#2 - 1") int x) {}

    @Positive
  void test1() {
    @Positive
    int offset = -4;
    @Positive
    int x = 4;
    @Positive
    int[] f1 = new int[x - offset];
    @Positive
    method1(f1, offset, x);
    @Positive
  }

    @Positive
  void method2(int[] a, int offset, @LTLengthOf(value = "#1", offset = "#2 - 1") int x) {}

    @Positive
  void test2() {
    @Positive
    int offset = 4;
    @Positive
    int x = 4;
    @Positive
    int[] f1 = new int[x + offset];
    @Positive
    method2(f1, offset, x);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
