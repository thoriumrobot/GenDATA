// Source-based slice around line 193
// Method: <com.google.common.reflect.TypeToken: Class getRawType()>

   *   <li>If {@code T} is a {@code Class} itself, {@code T} itself is returned.
   *   <li>If {@code T} is a {@link ParameterizedType}, the raw type of the parameterized type is
   *       returned.
   *   <li>If {@code T} is a {@link GenericArrayType}, the returned type is the corresponding array
   *       class. For example: {@code List<Integer>[] => List[]}.
   *   <li>If {@code T} is a type variable or a wildcard type, the raw type of the first upper bound
   *       is returned. For example: {@code <X extends Foo> => Foo}.
   * </ul>
   */
  public final Class<? super T> getRawType() {
    if (runtimeType instanceof Class) {
      @SuppressWarnings("unchecked") // raw type is T
      Class<? super T> result = (Class<? super T>) runtimeType;
      return result;
    } else if (runtimeType instanceof ParameterizedType) {
      @SuppressWarnings("unchecked") // raw type is |T|
      Class<? super T> result = (Class<? super T>) ((ParameterizedType) runtimeType).getRawType();
      return result;
    } else {
      // For a wildcard or type variable, the first bound determines the runtime type.
