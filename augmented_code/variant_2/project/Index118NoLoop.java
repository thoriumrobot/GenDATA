/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;

    @Positive
public class Index118NoLoop {

    @Positive
  public static void foo(String @ArrayLen(4) [] args, int i) {
    @Positive
    if (i >= 1 && i <= 3) {
    @Positive
      System.out.println(args[i]);
    @Positive
    }
    @Positive
  }
    @Positive
}
