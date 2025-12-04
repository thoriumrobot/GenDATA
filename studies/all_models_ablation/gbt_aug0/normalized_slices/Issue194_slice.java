import org.checkerframework.checker.index.qual.SameLen;

    @Positive
public class Issue194 {
    @Positive
  class Custom {
    @Positive
    public @LengthOf("this") int length() {
    @Positive
      throw new RuntimeException();
    @Positive
    }

    @Positive
    public Object get(@IndexFor("this") int i) {
    @Positive
      return null;
    @Positive
    }

    @Positive
