// Source-based slice around line 105
// Method: <com.google.common.base.Enums: Converter stringConverter(Class)>


  /**
   * Returns a serializable converter that converts between strings and {@code enum} values of type
   * {@code enumClass} using {@link Enum#valueOf(Class, String)} and {@link Enum#name()}. The
   * converter will throw an {@code IllegalArgumentException} if the argument is not the name of any
   * enum constant in the specified enum.
   *
   * @since 16.0
   */
  public static <T extends Enum<T>> Converter<String, T> stringConverter(Class<T> enumClass) {
    return new StringConverter<>(enumClass);
  }

  private static final class StringConverter<T extends Enum<T>> extends Converter<String, T>
      implements Serializable {

    private final Class<T> enumClass;

    StringConverter(Class<T> enumClass) {
      this.enumClass = checkNotNull(enumClass);
