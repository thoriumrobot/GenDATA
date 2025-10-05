    @Positive
package fieldinvar;

    @Positive
import org.checkerframework.common.value.qual.BottomVal;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.common.value.qual.MinLenFieldInvariant;
    @Positive
import org.checkerframework.framework.qual.FieldInvariant;

    @Positive
public class MinLenFieldInvar {
    @Positive
  class Super {
    @Positive
    public final int @MinLen(2) [] minlen2;

    @Positive
    public Super(int @MinLen(2) [] minlen2) {
    @Positive
      this.minlen2 = minlen2;
    @Positive
    }
    @Positive
  }

  // :: error: (field.invariant.not.subtype)
    @Positive
  class InvalidSub extends Super {
    @Positive
    public InvalidSub() {
    @Positive
      super(new int[] {1, 2});
    @Positive
    }
    @Positive
  }

    @Positive
  class ValidSub extends Super {
    @Positive
    public final int[] validSubField;

    @Positive
    public ValidSub(int[] validSubField) {
    @Positive
      super(new int[] {1, 2, 3, 4});
    @Positive
      this.validSubField = validSubField;
    @Positive
    }
    @Positive
  }

  // :: error: (field.invariant.not.found.superclass)
    @Positive
  class InvalidSubSub1 extends ValidSub {
    @Positive
    public InvalidSubSub1() {
    @Positive
      super(new int[] {1, 2});
    @Positive
    }
    @Positive
  }

  // :: error: (field.invariant.not.subtype.superclass)
    @Positive
  class InvalidSubSub2 extends ValidSub {
    @Positive
    public InvalidSubSub2() {
    @Positive
      super(new int[] {1, 2});
    @Positive
    }
    @Positive
  }

    @Positive
  @FieldInvariant(field = "minlen2", qualifier = BottomVal.class)
    @Positive
  class ValidSubSub extends ValidSub {
    @Positive
    public ValidSubSub() {
    @Positive
      super(null);
    @Positive
    }

    @Positive
    void test() {
    @Positive
      int @BottomVal [] bot = minlen2;
    @Positive
      int @MinLen(4) [] four = validSubField;
    @Positive
    }
    @Positive
  }
    @Positive
}

// CFWR semantic augmentation - variant 0
