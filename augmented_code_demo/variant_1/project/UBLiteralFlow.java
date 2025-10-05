    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrLow;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;

    @Positive
public class UBLiteralFlow {

    @Positive
  private static @IndexOrLow("#1") int lineStartIndexPartial(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf("#1") int lineStartIndexPartial2(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    // :: error: (return)
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    @Positive
    if (s.length() == 0) {
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @LTLengthOf("#1") int lineStartIndex2(String s, int start) {
    @Positive
    if (s.length() == 0) {
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndex3(String s, int start) {
    @Positive
    if (s.length() == 0) {
      // :: error: (return)
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      // :: error: (return)
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
      // :: error: (return)
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndex4(String s, int start) {
    @Positive
    if (s.length() == 0) {
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
