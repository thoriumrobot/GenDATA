    @Positive
    void minLenUse(int[] b) {
    @Positive
      minLenContract(b);
    @Positive
      int @MinLen(10) [] c = b;
    @Positive
    }

    @Positive
    public int b, y;

    @Positive
        expression = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "10"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int c) {
    @Positive
      if (b < a.length - c - 1 && b < a.length - 10) {
    @Positive
        return true;
    @Positive
      } else {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }

    // :: error: (flowexpr.parse.error)
    @Positive
    boolean ltlPostInvalid(int[] a, int c) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "-10"})
    @Positive
    void ltlPre(int[] a, int c) {
    @Positive
      @LTLengthOf(value = "a", offset = "1 + c") int i = b;
    @Positive
    }

    @Positive
    void ltlUse(int[] a, int c) {
    @Positive
      if (ltlPost(a, c)) {
    @Positive
        @LTLengthOf(value = "a", offset = "1 + c") int i = b;

    @Positive
        ltlPre(a, c);
    @Positive
      }
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1 + c") /*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.framework.qual.ConditionalPostconditionAnnotation;
    @Positive
import org.checkerframework.framework.qual.JavaExpression;
    @Positive
import org.checkerframework.framework.qual.PostconditionAnnotation;
    @Positive
import org.checkerframework.framework.qual.PreconditionAnnotation;
    @Positive
import org.checkerframework.framework.qual.QualifierArgument;

    @Positive
public class CustomContractWithArgs {
  // Postcondition for MinLen
    @Positive
  @PostconditionAnnotation(qualifier = MinLen.class)
    @Positive
  @interface EnsuresMinLen {
    @Positive
    public String[] value();

    @Positive
    public int targetValue();
    @Positive
  }

  // Conditional postcondition for LTLengthOf
    @Positive
  @ConditionalPostconditionAnnotation(qualifier = LTLengthOf.class)
    @Positive
  @interface EnsuresLTLIf {
    @Positive
    public boolean result();

    @Positive
    public String[] expression();

    @Positive
    public String[] targetValue();

    @Positive
    public String[] targetOffset();
    @Positive
  }

  // Precondition for LTLengthOf
    @Positive
  @PreconditionAnnotation(qualifier = LTLengthOf.class)
    @Positive
  @interface RequiresLTL {
    @Positive
    public String[] value();

    @Positive
    public String[] targetValue();

    @Positive
    public String[] targetOffset();
    @Positive
  }

    @Positive
  class Base {
    @Positive
    void minLenContract(int[] a) {
    @Positive
      if (a.length < 10) throw new RuntimeException();
    @Positive
    }

    // :: error: (contracts.postcondition)
    @Positive
    void minLenWrong(int[] a) {
    @Positive
      if (a.length < 9) throw new RuntimeException();
    @Positive
    }

    @Positive
    void minLenUse(int[] b) {
    @Positive
      minLenContract(b);
    @Positive
      int @MinLen(10) [] c = b;
    @Positive
    }

    @Positive
    public int b, y;

    @Positive
        expression = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "10"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int c) {
    @Positive
      if (b < a.length - c - 1 && b < a.length - 10) {
    @Positive
        return true;
    @Positive
      } else {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }

    // :: error: (flowexpr.parse.error)
    @Positive
    boolean ltlPostInvalid(int[] a, int c) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "-10"})
    @Positive
    void ltlPre(int[] a, int c) {
    @Positive
      @LTLengthOf(value = "a", offset = "1 + c") int i = b;
    @Positive
    }

    @Positive
    void ltlUse(int[] a, int c) {
    @Positive
      if (ltlPost(a, c)) {
    @Positive
        @LTLengthOf(value = "a", offset = "1 + c") int i = b;

    @Positive
        ltlPre(a, c);
    @Positive
      }
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1 + c") int b = b;
    @Positive
    }
    @Positive
  }

    @Positive
  class Derived extends Base {
    @Positive
    public int x;

    @Positive
        expression = "b ",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "11"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int d) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b ",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "-11"})
    @Positive
    void ltlPre(int[] a, int d) {
    @Positive
          value = {"a", "a"},
    @Positive
          offset = {"1 + d", "-10"})
      // :: error: (assignment)
    @Positive
      int i = b;
    @Positive
    }
    @Positive
  }

    @Positive
  class DerivedInvalid extends Base {
    @Positive
    public int x;

    @Positive
        expression = "b ",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "9"},
    @Positive
        result = true)
    // :: error: (contracts.conditional.postcondition.true.override)
    @Positive
    boolean ltlPost(int[] a, int c) {
      // :: error: (contracts.conditional.postcondition)
    @Positive
      return true;
    @Positive
    }

    @Positive
        value = "b ",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#1 + 2", "-9"})
    // :: error: (contracts.precondition.override)
    @Positive
    void ltlPre(int[] a, int d) {
    @Positive
          value = {"a", "a"},
    @Positive
          offset = {"1 + d", "-10"})
    @Positive
      int i = b;
    @Positive
    }
    @Positive
  }
    @Positive
}

    @Positive
    }
