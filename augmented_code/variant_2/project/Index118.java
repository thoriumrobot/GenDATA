/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class Index118 {

    @Positive
  public static void foo(String @ArrayLen(4) [] args) {
    @Positive
    for (int i = 1; i <= 3; i++) {
    @Positive
      @IntRange(from = 1, to = 3) int x = i;
    @Positive
      System.out.println(args[i]);
    @Positive
    }
    @Positive
  }

    @Positive
  public static void bar(@NonNegative int i, String @ArrayLen(4) [] args) {
    @Positive
    if (i <= 3) {
    @Positive
      System.out.println(args[i]);
    @Positive
    }
    @Positive
  }
    @Positive
}
