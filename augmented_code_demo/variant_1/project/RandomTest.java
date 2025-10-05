    @Positive
import java.util.Random;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class RandomTest {
    @Positive
  void test() {
    @Positive
    Random rand = new Random();
    @Positive
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @Positive
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @Positive
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @Positive
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
