// Tests handling Math.min and Math.max methods.
// The upper bound of Math.max is issue panacekcz#20:
// https://github.com/panacekcz/checker-framework/issues/20

    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;

    @Positive
public class MinMaxIndex {
  // Both min and max preserve IndexFor
    @Positive
  void indexFor(char[] array, @IndexFor("#1") int i1, @IndexFor("#1") int i2) {
    @Positive
    char c = array[Math.max(i1, i2)];
    @Positive
    char d = array[Math.min(i1, i2)];
    @Positive
  }

  // Both min and max preserve IndexOrHigh
    @Positive
  void indexOrHigh(String str, @IndexOrHigh("#1") int i1, @IndexOrHigh("#1") int i2) {
    @Positive
    str.substring(Math.max(i1, i2));
    @Positive
    str.substring(Math.min(i1, i2));
    @Positive
  }

  // Combining IndexFor and IndexOrHigh
    @Positive
  void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
    @Positive
    str.substring(Math.max(i1, i2));
    @Positive
    str.substring(Math.min(i1, i2));
    // :: error: (argument)
    @Positive
    str.charAt(Math.max(i1, i2));
    @Positive
    str.charAt(Math.min(i1, i2));
    @Positive
  }

  // max does not work with different sequences, min does
    @Positive
  void twoSequences(String str1, String str2, @IndexFor("#1") int i1, @IndexFor("#2") int i2) {
    // :: error: (argument)
    @Positive
    str1.charAt(Math.max(i1, i2));
    @Positive
    str1.charAt(Math.min(i1, i2));
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 1
