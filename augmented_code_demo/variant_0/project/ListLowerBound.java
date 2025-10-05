// @skip-test until we bring list support back
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;

    @Positive
public class ListLowerBound {

    @Positive
  private void m(List<Object> l) {
    // :: error: (argument)
    @Positive
    l.get(-1);
    // :: error: (argument)
    @Positive
    ListIterator<Object> li = l.listIterator(-1);

    @Positive
    @NonNegative int ni = li.nextIndex();
    @Positive
    @GTENegativeOne int pi = li.previousIndex();
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
