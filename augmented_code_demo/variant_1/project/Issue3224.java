// Test case for https://tinyurl.com/cfissue/3224

    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class Issue3224 {
    @Positive
  static class Arrays {
    @Positive
    static String[] copyOf(String[] args, int length) {
    @Positive
      return args;
    @Positive
    }
    @Positive
  }

    @Positive
  public static void m1(String @MinLen(1) [] args) {
    @Positive
    int i = 4;
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, i);
    @Positive
  }

    @Positive
  public static void m2(String @MinLen(1) [] args) {
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, args.length);
    @Positive
  }

    @Positive
  public static void m3(String @MinLen(1) ... args) {
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, args.length);
    @Positive
  }

    @Positive
  public static void m4(String @MinLen(1) [] args, @IntRange(from = 10, to = 200) int len) {
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, len);
    @Positive
  }

    @Positive
  public static void m5(String @MinLen(1) [] args, String[] otherArray) {
    // :: error: (assignment)
    @Positive
    String @MinLen(1) [] args2 = java.util.Arrays.copyOf(args, otherArray.length);
    @Positive
  }

    @Positive
  public static void m6(String @MinLen(1) [] args) {
    // :: error: (assignment)
    @Positive
    String @MinLen(1) [] args2 = Arrays.copyOf(args, args.length);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
