// Source-based slice around line 555
// Method: <com.google.common.base.Converter: Converter identity()>


    @Override
    public String toString() {
      return "Converter.from(" + forwardFunction + ", " + backwardFunction + ")";
    }
  }

  /** Returns a serializable converter that always converts or reverses an object to itself. */
  @SuppressWarnings("unchecked") // implementation is "fully variant"
  public static <T> Converter<T, T> identity() {
    return (IdentityConverter<T>) IdentityConverter.INSTANCE;
  }

  /**
   * A converter that always converts or reverses an object to itself. Note that T is now a
   * "pass-through type".
   */
  private static final class IdentityConverter<T> extends Converter<T, T> implements Serializable {
    static final Converter<?, ?> INSTANCE = new IdentityConverter<>();

