    @Positive
import org.checkerframework.checker.index.qual.LessThan;

    @Positive
public class LessThanFloat {
    @Positive
  int bigger;



    @Positive
  @LessThan("bigger") int i;


  // :: error: (anno.on.irrelevant)

  // :: error: (anno.on.irrelevant)

  // :: error: (anno.on.irrelevant)






  // :: error: (anno.on.irrelevant)

  // :: error: (anno.on.irrelevant)

  // :: error: (anno.on.irrelevant)


    @Positive
  java.lang.@LessThan("bigger") Byte bBoxed2;

    @Positive
  java.lang.@LessThan("bigger") Short sBoxed2;

    @Positive
  java.lang.@LessThan("bigger") Integer iBoxed2;

    @Positive
  java.lang.@LessThan("bigger") Long lBoxed2;

  // :: error: (anno.on.irrelevant)
    @Positive
  java.lang.@LessThan("bigger") Float fBoxed2;

  // :: error: (anno.on.irrelevant)
    @Positive
  java.lang.@LessThan("bigger") Double dBoxed2;

  // :: error: (anno.on.irrelevant)
    @Positive
  java.lang.@LessThan("bigger") Boolean boolBoxed2;

    @Positive
  java.lang.@LessThan("bigger") Character cBoxed2;
    @Positive
}

// CFWR semantic augmentation - variant 1
