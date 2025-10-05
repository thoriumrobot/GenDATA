// Source-based slice around line 29
// Method: <com.google.common.testing.anotherpackage.SomeClassThatDoesNotUseNullable: void staticButDoesNotCheckNull(String)>

@SuppressWarnings("unused") // For use by NullPointerTester
public class SomeClassThatDoesNotUseNullable {

  void packagePrivateButDoesNotCheckNull(String s) {}

  protected void protectedButDoesNotCheckNull(String s) {}

  public void publicButDoesNotCheckNull(String s) {}

  public static void staticButDoesNotCheckNull(String s) {}
}
