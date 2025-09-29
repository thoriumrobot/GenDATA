import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class Index118 {

    public static void bar(@NonNegative int i, String @ArrayLen(4) [] args) {
        if (i <= 3) {
            System.out.println(args[i]);
        }
    }
}
