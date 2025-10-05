/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
public class StringSameLen {
    @Positive
  public void m(String s) {
    @Positive
    String t = s;

    @Positive
    for (int i = 0; i < s.length(); ++i) {
    @Positive
      char c = t.charAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  public void m2(String s) {
    @Positive
    String t = s.toString();

    @Positive
    for (int i = 0; i < s.length(); ++i) {
    @Positive
      char c = t.charAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  public void m4(String s) {
    @Positive
    char[] t = s.toCharArray();

    @Positive
    for (int i = 0; i < s.length(); ++i) {
    @Positive
      char c = t[i];
    @Positive
    }
    @Positive
  }

    @Positive
  public void m6(char[] s) {
    @Positive
    String t = String.valueOf(s);

    @Positive
    for (int i = 0; i < s.length; ++i) {
    @Positive
      char c = t.charAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  public void m7(char[] s) {
    @Positive
    String t = String.copyValueOf(s);

    @Positive
    for (int i = 0; i < s.length; ++i) {
    @Positive
      char c = t.charAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  public void m8(String s) {
    @Positive
    String t = s.intern();

    @Positive
    for (int i = 0; i < s.length(); ++i) {
    @Positive
      char c = t.charAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  public void constructor(String s) {
    @Positive
    String t = new String(new char[] {'a'});
    @Positive
  }
    @Positive
}
