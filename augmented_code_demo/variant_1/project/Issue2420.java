    @Positive
import org.checkerframework.checker.index.qual.*;
    @Positive
import org.checkerframework.common.value.qual.*;

    @Positive
public class Issue2420 {
    @Positive
  static void str(String argStr) {
    @Positive
    if (argStr.isEmpty()) {
    @Positive
      return;
    @Positive
    }
    @Positive
    if (argStr == "abc") {
    @Positive
      return;
    @Positive
    }
    // :: error: (argument)
    @Positive
    char c = "abc".charAt(argStr.length() - 1);
    // :: error: (argument)
    @Positive
    char c2 = "abc".charAt(argStr.length());
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
