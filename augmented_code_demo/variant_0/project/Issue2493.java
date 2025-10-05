// test case for issue 2493: http://tinyurl.com/cfissue/2493

    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class Issue2493 {
    @Positive
  public static void test(int a[], int @SameLen("#1") [] b) {
    @Positive
    for (@IndexOrHigh("b") int i = 0; i < a.length; i++) {}
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
