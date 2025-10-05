// Test case for issue #64: https://github.com/kelloggm/checker-framework/issues/64

    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class LengthTransfer2 {
    @Positive
  public static void main(String[] args) {
    @Positive
    if (args.length != 2) {
    @Positive
      System.err.println("Needs 2 arguments, got " + args.length);
    @Positive
      System.exit(1);
    @Positive
    }
    @Positive
    int limit = Integer.parseInt(args[0]);
    @Positive
    int period = Integer.parseInt(args[1]);
    @Positive
  }

    @Positive
  void m(String @ArrayLen(2) [] args) {
    @Positive
    int limit = Integer.parseInt(args[0]);
    @Positive
    int period = Integer.parseInt(args[1]);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
