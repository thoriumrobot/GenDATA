    @Positive
package index;

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;

    @Positive
public class IndexForTestLBC {
    @Positive
  int[] array = {0};

    @Positive
  void test1(@IndexFor("array") int i) {
    @Positive
    int x = this.array[i];
    @Positive
  }

    @Positive
  void callTest1(int x) {
    @Positive
    test1(0);
    @Positive
    test1(1);
    @Positive
    test1(2);
    @Positive
    test1(array.length);
    // :: error: (argument)
    @Positive
    test1(array.length - 1);
    @Positive
    if (array.length > x) {
      // :: error: (argument)
    @Positive
      test1(x);
    @Positive
    }

    @Positive
    if (array.length == x) {
    @Positive
      test1(x);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
