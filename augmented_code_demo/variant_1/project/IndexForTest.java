    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.common.value.qual.MinLen;

    @Positive
public class IndexForTest {
    @Positive
  int @MinLen(1) [] array = {0};

    @Positive
  void test1(@IndexFor("array") int i) {
    @Positive
    int x = array[i];
    @Positive
  }

    @Positive
  void callTest1(int x) {
    @Positive
    test1(0);
    // ::  error: (argument)
    @Positive
    test1(1);
    // ::  error: (argument)
    @Positive
    test1(2);
    // ::  error: (argument)
    @Positive
    test1(array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test1(array.length - 1);
    @Positive
    }

    @Positive
    test1(array.length - 1);

    // ::  error: (argument)
    @Positive
    test1(this.array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test1(this.array.length - 1);
    @Positive
    }

    @Positive
    test1(this.array.length - 1);

    @Positive
    if (this.array.length > x && x >= 0) {
    @Positive
      test1(x);
    @Positive
    }

    @Positive
    if (array.length == x) {
      // ::  error: (argument)
    @Positive
      test1(x);
    @Positive
    }
    @Positive
  }

    @Positive
  void test2(@IndexFor("this.array") int i) {
    @Positive
    int x = array[i];
    @Positive
  }

    @Positive
  void callTest2(int x) {
    @Positive
    test2(0);
    // ::  error: (argument)
    @Positive
    test2(1);
    // ::  error: (argument)
    @Positive
    test2(2);
    // ::  error: (argument)
    @Positive
    test2(array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test2(array.length - 1);
    @Positive
    }

    @Positive
    test2(array.length - 1);

    // ::  error: (argument)
    @Positive
    test2(this.array.length);

    @Positive
    if (array.length > 0) {
    @Positive
      test2(this.array.length - 1);
    @Positive
    }

    @Positive
    test2(this.array.length - 1);

    @Positive
    if (array.length == x && x >= 0) {
      // ::  error: (argument)
    @Positive
      test2(x);
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
