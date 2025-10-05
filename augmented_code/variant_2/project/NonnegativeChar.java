/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import java.util.ArrayList;
    @Positive
import org.checkerframework.checker.index.qual.LowerBoundBottom;
    @Positive
import org.checkerframework.checker.index.qual.PolyLowerBound;

    @Positive
public class NonnegativeChar {
    @Positive
  void foreach(char[] array) {
    @Positive
    for (char value : array) {}
    @Positive
  }

    @Positive
  char constant() {
    @Positive
    return Character.MAX_VALUE;
    @Positive
  }

    @Positive
  char conversion(int i) {
    @Positive
    return (char) i;
    @Positive
  }

    @Positive
  public void takeList(ArrayList<Character> z) {}

    @Positive
  public void passList() {
    @Positive
    takeList(new ArrayList<Character>());
    @Positive
  }

    @Positive
  static class CustomList extends ArrayList<Character> {}

    @Positive
  public void passCustomList() {
    @Positive
    takeList(new CustomList());
    @Positive
  }

    @Positive
  public @LowerBoundBottom char bottomLB(@LowerBoundBottom char c) {
    @Positive
    return c;
    @Positive
  }

    @Positive
  public @PolyLowerBound char polyLB(@PolyLowerBound char c) {
    @Positive
    return c;
    @Positive
  }
    @Positive
}
