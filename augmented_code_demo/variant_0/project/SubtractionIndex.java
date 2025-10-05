    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

// @skip-test until the type system is enriched so it can express either
//   * N = Grid.length and N-1 = Grid.length-1, or
//   * i < N and i <= N-1

    @Positive
public class SubtractionIndex {

  // Version without annotations
    @Positive
  public static void main(String[] args) {
    @Positive
    int N = 8;
    @Positive
    int[] grid = new int[N];
    @Positive
    for (int i = 0; i < N; i++) {
    @Positive
      System.out.println(grid[(N - 1) - i]);
    @Positive
    }
    @Positive
  }

  // Version with annotations
    @Positive
  public static void mainAnnotated(String[] args) {
    @Positive
    int N = 8;
    @Positive
    int @MinLen(8) [] grid = new int[N];
    @Positive
    @LTLengthOf("grid") int zero = 0;
    @Positive
    for (@LTLengthOf("grid") int i = zero; i < N; i++) {
    @Positive
      System.out.println(grid[(N - 1) - i]);
    @Positive
      System.out.println(grid[(N - i)]);
    @Positive
      System.out.println(grid[(N - i) - 1]);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
