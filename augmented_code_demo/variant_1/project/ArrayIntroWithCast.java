    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ArrayIntroWithCast<T> {

    @Positive
  void test(String[] a, String[] b) {
    @Positive
    Object result = new Object[a.length + b.length];
    @Positive
    System.arraycopy(a, 0, result, 0, a.length);
    @Positive
  }

    @Positive
  void test2(String[] a, String[] b) {
    @Positive
    T[] result = (T[]) new Object[a.length + b.length];
    @Positive
    System.arraycopy(a, 0, result, 0, a.length);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
