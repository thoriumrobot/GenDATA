/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
// Tests suport for index annotations applied to CharSequence and related indices.

    @Positive
import java.io.IOException;
    @Positive
import java.io.StringWriter;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.common.value.qual.StringVal;

    @Positive
public class CharSequenceTest {
  // Tests that minlen is correctly applied to CharSequence assigned from String, but not
  // StringBuilder
    @Positive
  void minLenCharSequence() {
    // :: error: (assignment)
    @Positive
    @MinLen(10) CharSequence sb = new StringBuilder("0123456789");
    @Positive
  }

  // Tests the subSequence method
    @Positive
  void testSubSequence() {
    // Local variable used because of https://github.com/kelloggm/checker-framework/issues/165
    @Positive
    String str = "0123456789";
    @Positive
    str.subSequence(5, 8);
    // :: error: (argument)
    @Positive
    str.subSequence(5, 13);
    @Positive
  }

  // Dummy method that takes a CharSequence and its index
    @Positive
  void sink(CharSequence cs, @IndexOrHigh("#1") int i) {}

  // Tests passing sequences as CharSequence
    @Positive
  void argumentPassing() {
    @Positive
    String s = "0123456789";
    @Positive
    sink(s, 8);
    @Positive
    StringBuilder sb = new StringBuilder("0123456789");
    // :: error: (argument)
    @Positive
    sink(sb, 8);
    @Positive
  }

  // Tests forwardning sequences as CharSequence
    @Positive
  void agumentForwarding(String s, @IndexOrHigh("#1") int i) {
    @Positive
    sink(s, i);
    @Positive
  }

  // Tests concatenation of CharSequence and String
    @Positive
  void concat() {
    @Positive
    CharSequence a = "a";
    @Positive
    @StringVal({"nullb", "ab"}) CharSequence ab = a + "b";
    @Positive
    sink(ab, 2);
    @Positive
  }

  // Tests that length retrieved from CharSequence can be used as an index
    @Positive
  void getLength(CharSequence cs, int i) {
    @Positive
    if (i >= 0 && i < cs.length()) {
    @Positive
      cs.charAt(i);
    @Positive
    }

    @Positive
    @IndexOrHigh("cs") int l = cs.length();
    @Positive
  }

    @Positive
  void testCharAt(CharSequence cs, int i, @IndexFor("#1") int j) {
    @Positive
    cs.charAt(j);
    @Positive
    cs.subSequence(j, j);
    // :: error: (argument)
    @Positive
    cs.charAt(i);
    // :: error: (argument)
    @Positive
    cs.subSequence(i, j);
    @Positive
  }

    @Positive
  void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    @Positive
    app.append(cs, i, i);
    // :: error: (argument)
    @Positive
    app.append(cs, 1, 2);
    @Positive
  }

    @Positive
  void testAppend(StringWriter app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    @Positive
    app.append(cs, i, i);
    // :: error: (argument)
    @Positive
    app.append(cs, 1, 2);
    @Positive
  }
    @Positive
}
