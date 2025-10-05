    @Positive
import java.util.List;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class OffsetExample {
    @Positive
  void example1(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example2(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example3(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example4(int[] a, int offset) {
    @Positive
    int max_index = a.length - offset;
    @Positive
    for (int i = 0; i < max_index; i++) {
    @Positive
      a[i + offset] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  void example5(int[] a, int offset) {
    @Positive
    for (int i = 0; i < a.length - offset; i++) {
    @Positive
      a[i + offset] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
    @Positive
    if (end > start) {
      // If start == 0, then end - start is end.  end might be equal to the length of a.  So
      // the array access might be too high.
      // ::error: (array.access.unsafe.high)
    @Positive
      a[end - start] = 0;
    @Positive
    }

    @Positive
    if (end > start) {
    @Positive
      a[end - start - 1] = 0;
    @Positive
    }
    @Positive
  }

    @Positive
  public static boolean isSubarray(Object[] a, Object[] sub, int a_offset) {
    @Positive
    int a_len = a.length - a_offset;
    @Positive
    int sub_len = sub.length;
    @Positive
    if (a_len < sub_len) {
    @Positive
      return false;
    @Positive
    }
    @Positive
    for (int i = 0; i < sub_len; i++) {
    @Positive
      if (sub[i] == a[a_offset + i]) {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }
    @Positive
    return true;
    @Positive
  }

    @Positive
  public void test2(int[] a, List<Object> b) {
    @Positive
    int b_size = b.size();
    @Positive
    Object[] result = new Object[a.length + b_size];
    @Positive
    for (int i = 0; i < b_size; i++) {
    @Positive
      result[i + a.length] = b.get(i);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
