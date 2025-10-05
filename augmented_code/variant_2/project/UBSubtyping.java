/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int 1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}


    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}

    @Positive
    @UpperBoundUnknown /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int a = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}


    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int a = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}


    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int 2 = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}


    @Positive
    @LTEqLengthOf({"arr", "arr2"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int test = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}

    @Positive
    @LTEqLengthOf({"arr"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int test = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }
    @Positive
}

    @Positive
    @LTEqLengthOf({"arr", "arr3"}) /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.UpperBoundUnknown;

    @Positive
public class UBSubtyping {
    @Positive
  int[] arr = {1};
    @Positive
  int[] arr2 = {1};
    @Positive
  int[] arr3 = {1};

    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int test = test;
    @Positive
  }
    @Positive
}

    @Positive
  }
    @Positive
}
