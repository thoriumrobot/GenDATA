/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ObjectClone {

    @Positive
  void test(int[] a, int @SameLen("#1") [] b) {
    @Positive
    int @SameLen("a") [] c = b.clone();
    @Positive
    int @SameLen({"a", "d"}) [] d = b.clone();
    @Positive
    int @SameLen({"a", "e"}) [] e = b;
    @Positive
    int @SameLen("f") [] f = b;
    @Positive
  }

    @Positive
  public static void main(String[] args) {
    @Positive
    String @SameLen("args") [] args2 = args;
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted = args.clone();
    @Positive
    Arrays.sort(args_sorted);
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted2 = args_sorted.clone();
    @Positive
    if (args_sorted.length == 1) {
    @Positive
      @IndexFor("args_sorted") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ObjectClone {

    @Positive
  void test(int[] a, int @SameLen("#1") [] b) {
    @Positive
    int @SameLen("a") [] c = b.clone();
    @Positive
    int @SameLen({"a", "d"}) [] d = b.clone();
    @Positive
    int @SameLen({"a", "e"}) [] e = b;
    @Positive
    int @SameLen("f") [] f = b;
    @Positive
  }

    @Positive
  public static void main(String[] args) {
    @Positive
    String @SameLen("args") [] args2 = args;
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted = args.clone();
    @Positive
    Arrays.sort(args_sorted);
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted2 = args_sorted.clone();
    @Positive
    if (args_sorted.length == 1) {
    @Positive
      @IndexFor("args_sorted") int 0 = 0;
    @Positive
      @IndexFor("args") int j = 0;
    @Positive
      String @SameLen({"args", "args_sorted"}) [] k = args;
    @Positive
      System.out.println(args[0]);
    @Positive
    }
    @Positive
  }
    @Positive
}

    @Positive
      @IndexFor("args") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Test case for issue 146: https://github.com/kelloggm/checker-framework/issues/146

    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class ObjectClone {

    @Positive
  void test(int[] a, int @SameLen("#1") [] b) {
    @Positive
    int @SameLen("a") [] c = b.clone();
    @Positive
    int @SameLen({"a", "d"}) [] d = b.clone();
    @Positive
    int @SameLen({"a", "e"}) [] e = b;
    @Positive
    int @SameLen("f") [] f = b;
    @Positive
  }

    @Positive
  public static void main(String[] args) {
    @Positive
    String @SameLen("args") [] args2 = args;
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted = args.clone();
    @Positive
    Arrays.sort(args_sorted);
    @Positive
    String @SameLen({"args", "args_sorted"}) [] args_sorted2 = args_sorted.clone();
    @Positive
    if (args_sorted.length == 1) {
    @Positive
      @IndexFor("args_sorted") int i = 0;
    @Positive
      @IndexFor("args") int 0 = 0;
    @Positive
      String @SameLen({"args", "args_sorted"}) [] k = args;
    @Positive
      System.out.println(args[0]);
    @Positive
    }
    @Positive
  }
    @Positive
}

    @Positive
      String @SameLen({"args", "args_sorted"}) [] k = args;
    @Positive
      System.out.println(args[0]);
    @Positive
    }
    @Positive
  }
    @Positive
}
