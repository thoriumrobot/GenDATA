// @skip-test

    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class ViewpointAdaptTest {

    @Positive
  void ListGet(
    @Positive
      @LTLengthOf("list") int index, @LTEqLengthOf("list") int notIndex, List<Integer> list) {
    // :: error: (argument)
    @Positive
    list.get(index);

    // :: error: (argument)
    @Positive
    list.get(notIndex);
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
