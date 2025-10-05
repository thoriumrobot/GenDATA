    @Positive
import java.util.Random;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class RandomTestLBC {
    @Positive
  void test() {
    @Positive
    Random rand = new Random();
    @Positive
    int[] a = new int[8];
    // Math.random() and Math.nextDouble() are always non-negative, but the Index Checker
    // does not reason about floating-point values.
    // :: error: (anno.on.irrelevant)
    // :: error: (assignment)
    @Positive
    @NonNegative int deref = (int) (Math.random() * a.length);
    // :: error: (assignment)
    @Positive
    @NonNegative int deref2 = (int) (rand.nextDouble() * a.length);
    @Positive
    @NonNegative int deref3 = rand.nextInt(a.length);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
