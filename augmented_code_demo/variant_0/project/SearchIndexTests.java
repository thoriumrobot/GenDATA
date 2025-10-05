    @Positive
import java.util.Arrays;
    @Positive
import org.checkerframework.checker.index.qual.*;

    @Positive
public class SearchIndexTests {
    @Positive
  public void test(short[] a, short instant) {
    @Positive
    int i = Arrays.binarySearch(a, instant);
    @Positive
    @SearchIndexFor("a") int z = i;
    // :: error: (assignment)
    @Positive
    @SearchIndexFor("a") int y = 7;
    @Positive
    @LTLengthOf("a") int x = i;
    @Positive
  }

    @Positive
  void test2(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (0 > xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (-1 >= xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz < 0) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test5(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz <= -1) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void subtyping1(
    @Positive
      @SearchIndexFor({"#3", "#4"}) int x, @NegativeIndexFor("#3") int y, int[] a, int[] b) {
    // :: error: (assignment)
    @Positive
    @SearchIndexFor({"a", "b"}) int z = y;
    @Positive
    @SearchIndexFor("a") int w = y;
    @Positive
    @SearchIndexFor("b") int p = x;
    // :: error: (assignment)
    @Positive
    @NegativeIndexFor({"a", "b"}) int q = x;
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
