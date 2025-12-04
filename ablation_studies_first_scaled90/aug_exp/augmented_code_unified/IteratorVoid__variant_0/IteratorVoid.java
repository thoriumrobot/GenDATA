    @Positive
import java.util.Iterator;

// If quals are configured incorrectly, there will be an incompatible assignment error; this ensures
// that Void is given the Positive type.

    @Positive
public class IteratorVoid<T> {
    @Positive
  T next1;
    @Positive
  Iterator<T> itor1;

    @Positive
  private void setnext1() {
    @Positive
    next1 = itor1.hasNext() ? itor1.next() : null;
    @Positive
  }
    @Positive
}
