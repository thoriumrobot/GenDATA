/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int a = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int a = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int a = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int a = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int a = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int a = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int j = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int j = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int j = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int j = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int a = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int a = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int a = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int a = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    } else {
    @Positive
      @NonNegative /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int a = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int a = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int j = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int j = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    } else {
    @Positive
      @Positive /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int j = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int j = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int s = s;
    @Positive
    } else {
    @Positive
      @Positive int u = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    } else {
    @Positive
      @Positive /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public class RefinementGTE {

    @Positive
  void test_forward(int a, int j, int s) {
    /** forwards greater than or equals */
    // :: error: (assignment)
    @Positive
    @GTENegativeOne int aa = a;
    @Positive
    if (a >= -1) {
    @Positive
      @GTENegativeOne int b = a;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @GTENegativeOne int c = a;
    @Positive
    }

    @Positive
    if (j >= 0) {
    @Positive
      @NonNegative int k = j;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @NonNegative int l = j;
    @Positive
    }
    @Positive
  }

    @Positive
  void test_backwards(int a, int j, int s) {
    /** backwards greater than or equal */
    // :: error: (assignment)
    @Positive
    @NonNegative int aa = a;
    @Positive
    if (-1 >= a) {
      // :: error: (assignment)
    @Positive
      @NonNegative int b = a;
    @Positive
    } else {
    @Positive
      @NonNegative int c = a;
    @Positive
    }

    @Positive
    if (0 >= j) {
      // :: error: (assignment)
    @Positive
      @Positive int k = j;
    @Positive
    } else {
    @Positive
      @Positive int l = j;
    @Positive
    }

    @Positive
    if (1 >= s) {
      // :: error: (assignment)
    @Positive
      @Positive int t = s;
    @Positive
    } else {
    @Positive
      @Positive int s = s;
    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment

    @Positive
    }
    @Positive
  }
    @Positive
}
// a comment
