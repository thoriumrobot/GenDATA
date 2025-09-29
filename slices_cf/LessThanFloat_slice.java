  @LessThan("bigger") double d;

  // :: error: (anno.on.irrelevant)
  @LessThan("bigger") boolean bool;

  @LessThan("bigger") char c;

  @LessThan("bigger") Byte bBoxed;

  @LessThan("bigger") Short sBoxed;

  @LessThan("bigger") Integer iBoxed;

  @LessThan("bigger") Long lBoxed;

  // :: error: (anno.on.irrelevant)
  @LessThan("bigger") Float fBoxed;

  // :: error: (anno.on.irrelevant)
  @LessThan("bigger") Double dBoxed;

