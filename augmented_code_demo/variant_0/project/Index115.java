    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class Index115 {

    @Positive
  public static void main(String[] args) {
    @Positive
    if ((args.length > 1) && (args[1].equals("foo"))) {
    @Positive
      System.out.println("First argument is foo");
    @Positive
    }
    @Positive
  }

    @Positive
  public static void main2(String... args) {
    @Positive
    if ((args.length > 1) && (args[1].equals("foo"))) {
    @Positive
      System.out.println("First argument is foo");
    @Positive
    }
    @Positive
  }

    @Positive
  public static void main3(String @ArrayLen({1, 2}) [] args) {
    @Positive
    if ((args.length > 1) && (args[1].equals("foo"))) {
    @Positive
      System.out.println("First argument is foo");
    @Positive
    }
    @Positive
  }

    @Positive
  public static void main4(String @ArrayLen({1, 2}) ... args) {
    @Positive
    if ((args.length > 1) && (args[1].equals("foo"))) {
    @Positive
      System.out.println("First argument is foo");
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
